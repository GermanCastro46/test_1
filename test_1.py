import bladerf
from bladerf import _bladerf
from configparser import ConfigParser
import sys
import os
import threading
from multiprocessing.pool import ThreadPool


def shutdown( error = 0, board = None ):
    print( "Shutting down with error code: " + str(error) )
    if( board != None ):
        board.close()
    sys.exit(error)

def print_versions( device = None ):
    print( "libbladeRF version: " + str(_bladerf.version()) )
    if( device != None ):
        try:
            print( "Firmware version: " + str(device.get_fw_version()) )
        except:
            print( "Firmware version: " + "ERROR" )
            raise

        try:
            print( "FPGA version: "     + str(device.get_fpga_version()) )
        except:
            print( "FPGA version: "     + "ERROR" )
            raise

    return 0

def probe_bladerf():
    device = None
    print( "Searching for bladeRF devices..." )
    try:
        devinfos = _bladerf.get_device_list()
        if( len(devinfos) == 1 ):
            device = "{backend}:device={usb_bus}:{usb_addr}".format(**devinfos[0]._asdict())
            print( "Found bladeRF device: " + str(device) )
        if( len(devinfos) > 1 ):
            print( "Unsupported feature: more than one bladeRFs detected." )
            print( "\n".join([str(devinfo) for devinfo in devinfos]) )
            shutdown( error = -1, board = None )
    except _bladerf.BladeRFError:
        print( "No bladeRF devices found." )
        pass

    return device

def load_fpga( device, image ):

    image = os.path.abspath( image )

    if( not os.path.exists(image) ):
        print( "FPGA image does not exist: " + str(image) )
        return -1

    try:
        print( "Loading FPGA image: " + str(image ) )
        device.load_fpga( image )
        fpga_loaded  = device.is_fpga_configured()
        fpga_version = device.get_fpga_version()

        if( fpga_loaded ):
            print( "FPGA successfully loaded. Version: " + str(fpga_version) )

    except _bladerf.BladeRFError:
        print( "Error loading FPGA." )
        raise

    return 0

def receive(device, channel : int, freq : int, rate : int, gain : int,
            tx_start = None, rx_done = None,
            rxfile : str = '', num_samples : int = 1024):

    print("RX initializing....")
    print("Freq: " + str(freq))
    status = 0

    if( device == None ):
        print( "RX: Invalid device handle." )
        return -1

    if( channel == None ):
        print( "RX: Invalid channel." )
        return -1

    # Configure BladeRF
    ch             = device.Channel(channel)
    #ch = device.Channel(bladerf.CHANNEL_RX(0))
    #ch.sample_rate = rate
    ch.frequency   = freq
    #ch.bandwidth   = 10e6
    ch.gain        = gain

    # Setup synchronous stream
    device.sync_config(layout         = _bladerf.ChannelLayout.RX_X1,
                       fmt            = _bladerf.Format.SC16_Q11,
                       num_buffers    = 16,
                       buffer_size    = 8192,
                       num_transfers  = 8,
                       stream_timeout = 3500)

    # Enable module
    print( "RX: Start" )
    ch.enable = True

    # Create receive buffer
    bytes_per_sample = 4
    buf = bytearray(1024*bytes_per_sample)
    num_samples_read = 0

    # Tell TX thread to begin
    """if( tx_start != None ):
        tx_start.set()"""

    # Save the samples
    with open(rxfile, 'wb') as outfile:
        while True:
            if num_samples > 0 and num_samples_read == num_samples:
                break
            elif num_samples > 0:
                num = min(len(buf)//bytes_per_sample,
                          num_samples-num_samples_read)
            else:
                num = len(buf)//bytes_per_sample

            # Read into buffer
            device.sync_rx(buf, num)
            num_samples_read += num

            # Write to file
            outfile.write(buf[:num*bytes_per_sample])

    # Disable module
    print( "RX: Stop" )
    ch.enable = False

    if( rx_done != None ):
        rx_done.set()

    print( "RX: Done" )

    return 0

config = ConfigParser()
config.read('txrx.ini')

verbosity = config.get('common', 'libbladerf_verbosity').upper()
if  ( verbosity == "VERBOSE" ):  _bladerf.set_verbosity( 0 )
elif( verbosity == "DEBUG" ):    _bladerf.set_verbosity( 1 )
elif( verbosity == "INFO" ):     _bladerf.set_verbosity( 2 )
elif( verbosity == "WARNING" ):  _bladerf.set_verbosity( 3 )
elif( verbosity == "ERROR" ):    _bladerf.set_verbosity( 4 )
elif( verbosity == "CRITICAL" ): _bladerf.set_verbosity( 5 )
elif( verbosity == "SILENT" ):   _bladerf.set_verbosity( 6 )
else:
    print( "Invalid libbladerf_verbosity specified in configuration file:",
           verbosity )
    shutdown( error = -1, board = None )

uut = probe_bladerf()

if( uut == None ):
    print( "No bladeRFs detected. Exiting." )
    shutdown( error = -1, board = None )

b          = _bladerf.BladeRF( uut )
board_name = b.board_name
fpga_size  = b.fpga_size

print("Board Name: " + board_name)

if( config.getboolean(board_name + '-load-fpga', 'enable') ):
    print( "Loading FPGA..." )
    try:
        status = load_fpga( b, config.get(board_name + '-load-fpga',
                                          'image_' + str(fpga_size) + 'kle' ) )
    except:
        print( "ERROR loading FPGA." )
        raise

    if( status < 0 ):
        print( "ERROR loading FPGA." )
        shutdown( error = status, board = b )
else:
    print( "Skipping FPGA load due to configuration setting." )

status = print_versions( device = b )

print("mode: ")
#print_versions()
#probe_bladerf()

rx_pool = ThreadPool(processes=1)

for s in [ss for ss in config.sections() if board_name + '-' in ss]:
    #print(s)
    if( s == board_name + "-load-fpga" ):
        # Don't re-loading the FPGA!
        continue
        
    if( config.getboolean(s, 'enable') ):
        print( "RUNNING" )

        if( s == board_name + '-rx' ):

            rx_ch   = _bladerf.CHANNEL_RX(config.getint(s, 'rx_channel'))
            rx_freq = int(config.getfloat(s, 'rx_frequency'))
            rx_rate = int(config.getfloat(s, 'rx_samplerate'))
            rx_gain = int(config.getfloat(s, 'rx_gain'))
            rx_ns   = int(config.getfloat(s, 'rx_num_samples'))
            rx_file = config.get(s, 'rx_file')

            # Make this blocking for now ...
            status = rx_pool.apply_async(receive,
                                         (),
                                         { 'device'        : b,
                                           'channel'       : rx_ch,
                                           'freq'          : rx_freq,
                                           'rate'          : rx_rate,
                                           'gain'          : rx_gain,
                                           'tx_start'      : None,
                                           'rx_done'       : None,
                                           'rxfile'        : rx_file,
                                           'num_samples'   : rx_ns
                                         }).get()
            if( status < 0 ):
                print( "Receive operation failed with error " + str(status) )

b.close()
print("Done....")
