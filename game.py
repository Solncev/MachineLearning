import numpy as np
import pygame

from my_cmeans import kinda_main as c_means
from my_kmeans import kinda_main as k_means
from dbscan import main as dbscan


def create_data(position, data):
    (x, y) = position
    coord = [x, y]
    data = np.append(data, [coord], axis=0)
    return data


def create_data_spray(position, data):
    (x, y) = position
    r = np.random.uniform(0, 20)
    phi = np.random.uniform(0, 2 * np.pi)
    coord = [x + r * np.cos(phi), y + r * np.sin(phi)]
    data = np.append(data, [coord], axis=0)
    return data


def main(method):
    data = np.empty((0, 2), dtype='f')

    radius = 2
    color = (0, 0, 255)
    thickness = 0

    bg_color = (255, 255, 255)
    (width, height) = (640, 480)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('K means')

    running = True
    pushing = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pushing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                pushing = False

        if pushing and np.random.randint(100) > 60:
            data = create_data_spray(pygame.mouse.get_pos(), data)

        screen.fill(bg_color)

        for i, d in enumerate(data):
            pygame.draw.circle(screen, color, (int(d[0]), int(d[1])), radius, thickness)

        pygame.display.flip()
    pygame.quit()

    # print(data.shape)
    # k_means(data, 3, '')
    # c_means(data, 3, '', cut=0.9)
    # c_means(data, 3, '', cut=0.9)
    # c_means(data, 3, '', cut=0.9)
    # dbscan(data, 30, 2)
    # da_fuq(data)



if __name__ == '__main__':
    main()
