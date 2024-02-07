import pygame

def play_piano_sound(notes):
    file_paths=[f"notes\{note}.wav" for note in notes]
    pygame.display.set_caption('')
    pygame.mixer.init()
    pygame.mixer.stop()
    if len(file_paths)>8:
        pygame.mixer.set_num_channels(len(file_paths))
    channels = [pygame.mixer.Channel(i) for i in range(len(file_paths))]
    for i, file_path in enumerate(file_paths):
        sound_effect = pygame.mixer.Sound(file_path)
        channels[i].play(sound_effect)