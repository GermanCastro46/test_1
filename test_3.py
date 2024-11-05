import bladerf

d = bladerf.BladeRF()
print(d)
ch = d.Channel(bladerf.CHANNEL_RX(0))
print(ch)
ch.frequency = 915000000
print(ch.frequency)
ch.sample_rate = 20000000
print(ch.sample_rate)