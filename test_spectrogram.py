import numpy as np
import matplotlib.pyplot as plt

# Parámetros de simulación
fs = 20e6  # Frecuencia de muestreo en Hz (1 MHz)
fc = 2.4e9  # Frecuencia de la señal en Hz (100 kHz)
duration = 0.1  # Duración en segundos
num_samples = int(fs * duration)  # Número total de muestras
print("Num Samples: "+str(num_samples))
# Generar un vector de tiempo
t = np.arange(num_samples) / fs

amplitude = 1  # Amplitud de la señal
signal_i = amplitude * np.cos(2 * np.pi * fc * t)  # Componente en fase (I)
signal_q = amplitude * np.sin(2 * np.pi * fc * t)  # Componente en cuadratura (Q)
signal = signal_i + 1j * signal_q  # Señal compleja I + jQ

# Añadir ruido Gaussiano para simular condiciones realistas
noise_amplitude = 0.1  # Amplitud del ruido
noise = noise_amplitude * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
signal_noisy = signal + noise

print("Signal size: "+str(signal_noisy.shape))
fft_size = 256
#num_rows = int(fs) // fft_size
num_rows = int(num_samples) // fft_size
print("Num Rows: "+str(num_rows))
spectrogram = np.zeros((num_rows, fft_size))
fft_vec = np.zeros((num_rows, fft_size))

for i in range(num_rows):
    fft_vec[i,:] = (np.fft.fft(signal_noisy[i*fft_size:(i+1)*fft_size]))
    spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(signal_noisy[i*fft_size:(i+1)*fft_size])))**2)
    #print("shape: "+str(spectrogram.shape))
fft_vec = np.abs(fft_vec)**2/100
total_pwr = np.sum(fft_vec)
print("Sum shape: "+str(fft_vec.shape))
print("Sum: "+str(total_pwr)) 
print("Total Power: ",str(10*np.log10(total_pwr*1000)))   
print("Rows done: "+str(i))
print("Spectrogram shape: "+str(spectrogram.shape))
"""plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:1000], signal_noisy.real[:1000], label="I (En Fase)")
plt.plot(t[:1000], signal_noisy.imag[:1000], label="Q (Cuadratura)")
plt.title("Simulación de Señal I/Q con Ruido")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.fft.fftfreq(num_samples, 1/fs), 20 * np.log10(np.abs(np.fft.fft(signal_noisy))))
plt.title("Espectro de Frecuencia")
plt.xlabel("Frecuencia (Hz)")"""

plt.figure()
#Pxx, freqs, bins, im = plt.specgram(signal_noisy, NFFT=1024, Fs=fs, Fc0, noverlap=512, cmap='viridis', sides='default', mode='default')
print("Min: "+str(spectrogram.min()))
print("Max: "+str(spectrogram.max()))
img = plt.imshow(spectrogram, aspect='auto', extent=[(fc + fs/-2)/1e9, (fc + fs/2)/1e9, 0.1, 0], origin='lower', cmap='viridis')
img.set_clim(spectrogram.min(),spectrogram.max())

plt.title('Espectrograma de la Señal Simulada')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Tiempo (s)')
plt.colorbar(label='Intensidad (dB)')
plt.ylim(0, 1)  # Limitar el eje y al rango de duración
#plt.ylim(0, fs/2)  # Limitar el rango de frecuencia a la mitad de la frecuencia de muestreo
plt.show()

print("Ending code.....")
