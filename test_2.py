import bladerf
from bladerf import _bladerf
from configparser import ConfigParser
import sys
import os
import threading
from multiprocessing.pool import ThreadPool
import numpy as np
import matplotlib.pyplot as plt

def shutdown( error = 0, board = None ):
    print( "Shutting down with error code: " + str(error) )
    if( board != None ):
        board.close()
    sys.exit(error)

def print_versions( device = None ):
    print( "libbladeRF version: " + str(bladerf.version()) )
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
        devinfos = bladerf.get_device_list()
        if( len(devinfos) == 1 ):
            #device = "{backend}:device={usb_bus}:{usb_addr}".format(**devinfos[0]._asdict())
            device = bladerf.BladeRF()
            print( "Found bladeRF device: " + str(device) )
        if( len(devinfos) > 1 ):
            print( "Unsupported feature: more than one bladeRFs detected." )
            print( "\n".join([str(devinfo) for devinfo in devinfos]) )
            shutdown( error = -1, board = None )
    except bladerf.BladeRFError:
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

    except bladerf.BladeRFError:
        print( "Error loading FPGA." )
        raise

    return 0

def receive(device, freq : int, rate : int, gain : int,
            tx_start = None, rx_done = None,
            rxfile : str = '', num_samples : int = 1024):

    print("RX initializing....")
    print("device " + str(device))
    status = 0

    if( device == None ):
        print( "RX: Invalid device handle." )
        return -1

    """if( channel == None ):
        print( "RX: Invalid channel." )
        return -1"""

    # Configure BladeRF
    ch= device.Channel(bladerf.CHANNEL_RX(0))
    print(ch)
    #ch = device.Channel(bladerf.CHANNEL_RX(0))
    ch.frequency   = freq
    print(ch.frequency)
    ch.sample_rate = rate
    #ch.bandwidth   = 2.5e6
    ch.gain_mode = _bladerf.GainMode.Manual
    ch.gain        = gain

    # Setup synchronous stream
    device.sync_config(layout         = _bladerf.ChannelLayout.RX_X1,
                       fmt            = _bladerf.Format.SC16_Q11,
                       num_buffers    = 16,
                       buffer_size    = 16384,
                       num_transfers  = 8,
                       stream_timeout = 3500)

    # Enable module
    print( "RX: Start" )
    ch.enable = True

    # Create receive buffer
    bytes_per_sample = 4
    buf = bytearray(1024*bytes_per_sample)
    num_samples_read = 0
    x = np.zeros(num_samples, dtype=np.complex64) 

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
            #print("Num samples readed: "+str(num))
            # Read into buffer
            device.sync_rx(buf, num)
            samples = np.frombuffer(buf, dtype=np.int16)
            samples = samples[0::2] + 1j * samples[1::2] # Convert to complex type
            samples /= 2048.0 # Scale to -1 to 1 (its using 12 bit ADC)
            x[num_samples_read:num_samples_read+num] = samples[0:num]
            num_samples_read += num

            # Write to file
            outfile.write(buf[:num*bytes_per_sample])

    # Disable module
    print( "RX: Stop" )
    ch.enable = False

    if( rx_done != None ):
        rx_done.set()

    print( "RX: Done" )

    fft_size = 2048
    num_rows = len(x) // fft_size # // is an integer division which rounds down
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
    extent = [(freq + rate/-2)/1e6, (freq + rate/2)/1e6, len(x)/rate, 0]
    plt.imshow(spectrogram, aspect='auto', extent=extent)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")
    plt.show()

    return 0

config = ConfigParser()
config.read('txrx.ini')

verbosity = config.get('common', 'libbladerf_verbosity').upper()
if  ( verbosity == "VERBOSE" ):  bladerf.set_verbosity( 0 )
elif( verbosity == "DEBUG" ):    bladerf.set_verbosity( 1 )
elif( verbosity == "INFO" ):     bladerf.set_verbosity( 2 )
elif( verbosity == "WARNING" ):  bladerf.set_verbosity( 3 )
elif( verbosity == "ERROR" ):    bladerf.set_verbosity( 4 )
elif( verbosity == "CRITICAL" ): bladerf.set_verbosity( 5 )
elif( verbosity == "SILENT" ):   bladerf.set_verbosity( 6 )
else:
    print( "Invalid libbladerf_verbosity specified in configuration file:",
           verbosity )
    shutdown( error = -1, board = None )

uut = probe_bladerf()

if( uut == None ):
    print( "No bladeRFs detected. Exiting." )
    shutdown( error = -1, board = None )

#b = bladerf.BladeRF( uut )
#b = bladerf.BladeRF()
b = uut
board_name = b.board_name
fpga_size  = b.fpga_size

print("Board Name: " + board_name)

"""if( config.getboolean(board_name + '-load-fpga', 'enable') ):
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
    print( "Skipping FPGA load due to configuration setting." )"""

#status = print_versions( device = b )

#rx_pool = ThreadPool(processes=1)

for s in [ss for ss in config.sections() if board_name + '-' in ss]:
    #print(s)
    if( s == board_name + "-load-fpga" ):
        # Don't re-loading the FPGA!
        continue
        
    if( config.getboolean(s, 'enable') ):
        print( "RUNNING" )

        if( s == board_name + '-rx' ):

            #rx_ch   = bladerf.CHANNEL_RX(config.getint(s, 'rx_channel'))
            #rx_ch = b.Channel(bladerf.CHANNEL_RX(0))
            rx_freq = int(config.getfloat(s, 'rx_frequency'))
            rx_rate = int(config.getfloat(s, 'rx_samplerate'))
            rx_gain = int(config.getfloat(s, 'rx_gain'))
            rx_ns   = int(config.getfloat(s, 'rx_num_samples'))
            rx_file = config.get(s, 'rx_file')

            # Make this blocking for now ...
            """status = rx_pool.apply_async(receive,
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
                                         }).get()"""
            receive(b,rx_freq,rx_rate,rx_gain,None,None,rx_file,rx_ns)
            """if( status < 0 ):
                print( "Receive operation failed with error " + str(status) )"""

b.close()
print("Done....")
#d = bladerf.BladeRF()
#print(d)
#ch = d.Channel(bladerf.CHANNEL_RX(0))
#print(ch)
#ch.frequency = 915000000
#print(ch.frequency)