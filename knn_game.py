import numpy as np
import pygame

from knn import main as knn


def create_data(position, data):
    (x, y) = position
    coord = [x, y]
    data = np.append(data, [coord], axis=0)
    return data


def create_data_spray(position, data, label):
    (x, y) = position
    r = np.random.uniform(0, 20)
    phi = np.random.uniform(0, 2 * np.pi)
    coord = [x + r * np.cos(phi), y + r * np.sin(phi)]
    data = np.append(data, [[coord, [label]]], axis=0)
    return data


def main(method):
    data = np.empty((0, 2), dtype='f')

    radius = 2
    color = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
    thickness = 0

    bg_color = (255, 255, 255)
    (width, height) = (640, 480)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('knn')

    running = True
    pushing = False
    label = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pushing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                pushing = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    label = 0
                elif event.key == pygame.K_1:
                    label = 1
                elif event.key == pygame.K_2:
                    label = 2

        if pushing and np.random.randint(100) > 60:
            data = create_data_spray(pygame.mouse.get_pos(), data, label)

        screen.fill(bg_color)

        for i, d in enumerate(data):
            pygame.draw.circle(screen, color[d[1][0]], (int(d[0][0]), int(d[0][1])), radius, thickness)

        pygame.display.flip()
    pygame.quit()


if __name__ == '__main__':
    main(knn)
    # pass