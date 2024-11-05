import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bladeRF import BladeRF

# Parámetros de la señal
fs = 10e6  # Frecuencia de muestreo
duration = 5  # Duración total en segundos
fft_size = 2048  # Tamaño de la ventana para la FFT
num_rows = int(fs * duration / fft_size)  # Número de filas en el espectrograma

# Inicializa el espectrograma
spectrogram = np.zeros((num_rows, fft_size))

# Crea la figura y los ejes
fig, ax = plt.subplots()
cax = ax.imshow(spectrogram, aspect='auto', origin='lower', 
                extent=[0, duration, 0, fs/2], cmap='viridis')
ax.set_title('Espectrograma Dinámico del bladeRF')
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Frecuencia (Hz)')
plt.colorbar(cax, label='Amplitud (dB)')

# Configuración del bladeRF
blade = BladeRF()
blade.open()  # Abre el dispositivo
blade.set_sample_rate(fs)  # Establece la frecuencia de muestreo
blade.set_frequency(2.4e9)  # Establece la frecuencia de operación
blade.set_gain(20)  # Establece la ganancia

# Función para actualizar el espectrograma
def update(frame):
    global spectrogram

    # Captura una ventana de datos
    segment = blade.receive(fft_size)  # Captura muestras del bladeRF

    # Actualiza el espectrograma
    spectrogram = np.roll(spectrogram, -1, axis=0)  # Desplaza las filas hacia arriba
    spectrum = np.fft.fftshift(np.fft.fft(segment))  # Calcular la FFT
    spectrogram[-1, :] = 10 * np.log10(np.abs(spectrum)**2)  # Guardar en dB

    # Actualiza la imagen
    cax.set_array(spectrogram.T)  # Transponer para que se ajuste a los ejes
    return cax,

# Configura la animación
ani = animation.FuncAnimation(fig, update, frames=100, blit=True, interval=100)

# Ejecutar la animación
plt.show()

# Cierra el dispositivo después de la visualización
blade.close()