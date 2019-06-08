#!/usr/bin/env python3

import random
import pygame
from tqdm import tqdm
from helpers import getDataDirectory, clearDirectory


WIDTH = 320
HEIGHT = 240


def getRandomColor():
	red = int(random.random() * 256)
	green = int(random.random() * 256)
	blue = int(random.random() * 256)
	return (red, green, blue)

# write to random files
def outputCircle(path):
	# darker circle on a dark background
	image = pygame.Surface((WIDTH, HEIGHT))
	size = int(random.random() * HEIGHT - 40)
	size = max(10, size)
	# draw the circle
	xpos = int(random.random() * (WIDTH - size))
	ypos = int(random.random() * (HEIGHT - size))
	image.fill((20, 20, 20))
	color = getRandomColor()
	color = (color[0] // 2, color[1] // 2, color[2] // 2)
	pygame.draw.circle(image, color, (xpos, ypos), int(size // 2))
	# save the image
	pygame.image.save(image, path)


def outputSquare(path):
	# square on a light background
	image = pygame.Surface((WIDTH, HEIGHT))
	size = int(random.random() * HEIGHT - 40)
	size = max(10, size)
	xpos = random.random() * (WIDTH - size)
	ypos = random.random() * (HEIGHT - size)
	image.fill((220, 220, 220))
	color = getRandomColor()
	area = pygame.Rect(xpos, ypos, size, int(size))
	pygame.draw.rect(image, color, area)
	pygame.image.save(image, path)


def output(dest, total, f):
	filenumber = 0
	dest = getDataDirectory(dest)
	for i in tqdm(range(total)):
		path = '{0}/{1}.png'.format(dest, filenumber)
		f(path)
		filenumber += 1

if __name__ == '__main__':
	for i in ['DATA/Train/GD', 'DATA/Valid/GD', 'DATA/Train/Other', 'DATA/Valid/Other']:
		clearDirectory(getDataDirectory(i))
	pygame.init()
	print('Creating 4 batches of images')
	output('DATA/Train/GD', 256, outputSquare)
	output('DATA/Train/Other', 256, outputCircle)
	output('DATA/Valid/GD', 128, outputSquare)
	output('DATA/Valid/Other', 128, outputCircle)
