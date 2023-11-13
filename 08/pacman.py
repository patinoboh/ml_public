# Import necessary modules
import pygame
from pygame.locals import *

# Define the Ghost class
class Ghost:
    def __init__(self, position, color):
        self.position = position
        self.color = color
        self.direction = (1, 0)  # Move right by default

    def move(self):
        # Update the position based on the current direction
        self.position = (self.position[0] + self.direction[0], self.position[1] + self.direction[1])


# Initialize the game
pygame.init()

# Set the screen size
SCREEN_SIZE = (600, 600)
screen = pygame.display.set_mode(SCREEN_SIZE)

# Set the title and icon
pygame.display.set_caption("Pac-Man")
icon = pygame.image.load("pacman_icon.png")
pygame.display.set_icon(icon)

# Set the background color
bg_color = (0, 0, 0)

# Create the ghosts
blinky = Ghost((300, 300), (255, 0, 0))  # red ghost
pinky = Ghost((300, 300), (255, 105, 180))  # pink ghost
inky = Ghost((300, 300), (0, 255, 255))  # cyan ghost
clyde = Ghost((300, 300), (255, 165, 0))  # orange ghost

# Store the ghosts in a list
ghosts = [blinky, pinky, inky, clyde]

# Game loop
running = True
while running:
    # Set the background color
    screen.fill(bg_color)

    # Check for user input
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
        elif event.type == QUIT:
            running = False

    # Update the game state

    # Update the ghosts' positions
    for ghost in ghosts:
        ghost.move()

    # Render the game

    # Render the ghosts
    for ghost in ghosts:
        pygame.draw.circle(screen, ghost.color, ghost.position, 20)

    # Update the screen
    pygame.display.flip()

# Clean up
pygame.quit()
