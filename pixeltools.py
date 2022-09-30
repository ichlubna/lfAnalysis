class PixelTools:
    def splitChannels(self, pixels):
        channels = []
        for channel in pixels[0]:
            channels.append([])
        for pixel in pixels:
            for i in range(len(pixel)):
                channels[i].append(pixel[i])
        return channels
