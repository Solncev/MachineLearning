import pygame


def main():
    bg_color = (255, 255, 255)
    (width, height) = (640, 480)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('K means')

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(bg_color)
        pygame.display.flip()
    pygame.quit()


if __name__ == '__main__':
    main()
