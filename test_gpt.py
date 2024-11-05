import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
import bladerf
from bladerf import _bladerf
import sys
from configparser import ConfigParser
import matplotlib.animation as animation

def shutdown( error = 0, board = None ):
    print( "Shutting down with error code: " + str(error) )
    if( board != None ):
        board.close()
    sys.exit(error)

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
    except _bladerf.BladeRFError:
        print( "No bladeRF devices found." )
        pass

    return device

# Bucle continuo de adquisición
def acquire_samples():
    x = np.zeros(int(rx_rate), dtype=np.complex64) 
    bytes_per_sample = 4
    buf = bytearray(1024*bytes_per_sample)
    num_samples_read = 0
    num_samples = int(rx_ns)
    while True:
        if num_samples > 0 and num_samples_read == num_samples:
            break
        elif num_samples > 0:
            num = min(len(buf)//bytes_per_sample,
            num_samples-num_samples_read)
        else:
            num = len(buf)//bytes_per_sample
        #print("Num: "+str(num))
        samples = uut.sync_rx(buf,num)  # Recibir 4096 muestras del BladeRF
        samples = np.frombuffer(buf, dtype=np.int16)
        samples = samples[0::2] + 1j * samples[1::2] # Convert to complex type
        samples /= 2048.0 # Scale to -1 to 1 (its using 12 bit ADC)
        x[num_samples_read:num_samples_read+num] = samples[0:num]
        num_samples_read += num
    return x

# Función para actualizar el espectrograma en cada intervalo
def update_spectrogram(x):
    global spectrogram
    global cax

    #print(fft_size)
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
    #extent = [(915e6 + 20e6/-2)/1e6, (915e6 + 20e6/2)/1e6, len(x)/20e6, 0]
    cax.set_data(spectrogram)
    cax.set_clim(vmin=spectrogram.min(), vmax=spectrogram.max())
    #plt.imshow(spectrogram, aspect='auto', extent=extent)
    return [cax]

def animate(i):
    # Adquirir nuevas muestras desde el BladeRF
    new_data = acquire_samples()

    # Actualizar el espectrograma con las nuevas muestras
    update_spectrogram(new_data)

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

for s in [ss for ss in config.sections() if board_name + '-' in ss]:
    if( config.getboolean(s, 'enable') ):
        if( s == board_name + '-rx' ):
            print( "RUNNING" )
            rx_freq = int(config.getfloat(s, 'rx_frequency'))
            rx_rate = int(config.getfloat(s, 'rx_samplerate'))
            rx_gain = int(config.getfloat(s, 'rx_gain'))
            rx_ns   = int(config.getfloat(s, 'rx_num_samples'))
            rx_file = config.get(s, 'rx_file')

ch= uut.Channel(bladerf.CHANNEL_RX(0))
#ch.frequency   = 2412e6
ch.frequency   = rx_freq
print(ch.frequency)
ch.sample_rate = rx_rate
#ch.bandwidth   = 2.5e6
ch.gain_mode = _bladerf.GainMode.Manual
ch.gain        = rx_gain
# Setup synchronous stream
uut.sync_config(layout         = _bladerf.ChannelLayout.RX_X1,
                   fmt            = _bladerf.Format.SC16_Q11,
                   num_buffers    = 16,
                   buffer_size    = 16384,
                   num_transfers  = 8,
                   stream_timeout = 3500)
#bytes_per_sample = 4
#buf = bytearray(1024*bytes_per_sample)
#num = len(buf)//bytes_per_sample

# Parámetros de la señal
fs = rx_rate  # Frecuencia de muestreo
duration = 1  # Duración total en segundos
fft_size = 2048  # Tamaño de la ventana para la FFT
#num_rows = int(fs * duration / fft_size)  # Número de filas en el espectrograma
num_rows = int(rx_rate) // fft_size
# Inicializa el espectrograma
spectrogram = np.zeros((num_rows, fft_size))

# Configuración del gráfico del espectrograma
fig, ax = plt.subplots()
cax = ax.imshow(spectrogram, aspect='auto', extent=[(rx_freq + rx_rate/-2)/1e6, (rx_freq + rx_rate/2)/1e6, 1, 0], cmap='jet')
ax.set_title('Espectrograma Dinámico del bladeRF')
ax.set_xlabel('Frecuencia (MHz)')
ax.set_ylabel('Tiempo (s)')
plt.colorbar(cax, label='Amplitud (dB)')

print("img shape: "+str(cax.get_array().shape))
# Enable module
print( "RX: Start" )
ch.enable = True
# Función de animación para actualizar el gráfico en tiempo real
ani = FuncAnimation(fig, animate, frames=200, interval=1000, blit=False)
#ani = animation.FuncAnimation(fig, animate, frames=None, interval=1500)
# Mostrar el gráfico
plt.show()
uut.close()