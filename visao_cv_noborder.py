'''
    - roteiro seguido: https://www.pucsp.br/~jarakaki/pai/Roteiro4.pdf
    - link rotacao: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    - python -m pip install --upgrade pip 
    - python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
    - pip install opencv-python
    - separei em funcoes, pq acho que assim da pra entender melhor oq cada coisa faz
        e nao ter que ficar memorisando funcao da biblioteca

'''
import os
from os import system
system("cls")

import cv2
import numpy as np
import math

PATH_BASE = "./bases/01 - completo/"
PATH_DEF = "./defeitos/01 - completo/"
N_IMAGENS_BASE = 1
N_IMAGENS_DEF = 3
INDEX_BASE = 0
INDEX_DEF = 1
RESIZE = 30
# RESIZE = 100

class Ponto:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_string(self):
        return("x: %i, y: %i" %(self.x, self.y))

def get_empty_img(img):
    h_1, w_1 = img.shape[:2]
    return np.zeros((h_1, w_1, 3), np.uint8)

def get_empty_img_grayscale(img):
    h_1, w_1 = img.shape[:2]
    return np.zeros((h_1, w_1, 1), np.uint8)

def copia_colorida(img):
    try:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    except:
        return img

def carrega_img_base(dir_name):
    path, dirs, files = next(os.walk("./bases/" + dir_name))
    
    imagens = []
    for file_name in files:
        imagens.append(cv2.imread(path + file_name, cv2.IMREAD_GRAYSCALE))
    return imagens

def carrega_img_def(dir_name):
    path, dirs, files = next(os.walk("./defeitos/" + dir_name))
    imagens = []
    for file_name in files:
        imagens.append(cv2.imread(path + file_name, cv2.IMREAD_GRAYSCALE))
    return imagens

def show_img(img, name='image', delay = 0):
    result = img
    
    # name='image'    #   Comment if u wanto to rename windows

    cv2.imshow(name, cv2.resize(result, (1024, 640), interpolation= cv2.INTER_AREA))
    if delay == 0: print("============================\n\tpress enter\n============================\n")
    cv2.waitKey(delay)

def mostra_pontos(img, pontos, delay=0):
    height, width = img.shape[:2]
    color =  copia_colorida(img)

    for ponto in pontos:
        for y in range(ponto.y - 2, ponto.y + 3):
            for x in range(ponto.x - 2, ponto.x + 3):
                if y >= height or x >= width: continue
                color[y][x] = [0,0, 255]

    show_img(color, "progress", delay)
    return color

def linha_vertical(img, ponto, show_progress = False, delay = 0):
    height, width = img.shape[:2]
    color = copia_colorida(img)

    for y in range(height):
        color[y][ponto.x] = [0, 255, 0]
        if show_progress and y % int(height/20) == 0: show_img(color, "progress", delay)
    
    show_img(color, "progress")
    return color

def linha_horizontal(img, ponto, show_progress = False, delay = 0):
    height, width = img.shape[:2]
    color = copia_colorida(img)

    for x in range(width):
        color[ponto.y][x] = [255, 0, 0]
        if show_progress and x % int(width/10) == 0: show_img(color, "progress", delay)
    
    # show_img(color, "progress")
    return color

def get_pixel_mais_acima(img):
    height, width = img.shape[:2]

    off_cima = 0
    for y in range(height):
        for x in range(width):
            if img[y][x] != 255: return Ponto(x, y)

def get_pixel_mais_a_esquerda(img):
    height, width = img.shape[:2]

    off_cima = 0
    for x in range(width):
        for y in range(height):
            if img[y][x] != 255: return Ponto(x, y)

def remove_bordas(img, show_progress = False, delay = 0):
    height, width = img.shape[:2]

    off_cima = get_pixel_mais_acima(img)
    off_esquerda = get_pixel_mais_a_esquerda(img)
    print(off_cima.to_string())
    print(off_esquerda.to_string())
    color = mostra_pontos(img, [off_cima, off_esquerda])
    color = linha_vertical(color, off_esquerda, show_progress, delay)
    color = linha_horizontal(color, off_cima, show_progress, delay)
    color = mostra_pontos(color, [Ponto(off_esquerda.x, off_cima.y)])

    copy = get_empty_img_grayscale(img)

    for y in range(height):
        py = y + off_cima.y
        if py >= height: break
        if show_progress and y % int(height / 20) == 0:
            show_img(copy, "progress", delay)
        for x in range(width):
            px = x + off_esquerda.x
            if px >= width: break

            copy[y][x] = img[py][px]
    
    show_img(copy, "progress")
    
    pass

def satura(img, show_progress = False, delay = 0):
    height, width = img.shape[:2]
    
    if show_progress: 
        show_img(img, "progress")
        print("SATURANDO IMAGEM")


    for py in range(0, height):
        if show_progress:
            if py % int(height/20) == 0: 
                # show_img(img, "progress", delay)
                linha_horizontal(img, Ponto(0, py), show_progress, delay)

            # if py % 30 == 0: show_img(img, "progress", delay)
        for px in range(0,  width):
            pixel = int(img[py][px] / 50) * 50
            if pixel >= 200: pixel = 255
            if pixel <= 50 : pixel = 0
            img[py][px] = pixel
    
    if show_progress:
        show_img(img, "progress", delay)

    r = 63
    img = np.uint8(img/r) * r
    
    not_colored = []
    for py in range(0, height):
        if show_progress:
            if py % int(height/20) == 0: show_img(img, "progress", delay)
            # if py % 30 == 0: show_img(img, "progress", delay)
        for px in range(0,  width):
            if img[py][px] >= 200:  img[py][px] = 255
            elif img[py][px] <= 50:  img[py][px] = 0
            elif img[py][px] > 50 and img[py][px] <= 130: img[py][px] = 100
            elif img[py][px] > 130 and img[py][px] < 200: img[py][px] = 150
            elif img[py][px] not in not_colored: not_colored.append(img[py][px])


    if show_progress: 
        print("IMAGEM SATURADA")
        show_img(img, "progress")

    return img

def remove_pixel_isolado(img, show_progress = False, delay = 0):
    height, width = img.shape[:2]
    for y in range(height):
        count = 0
        for x in range(width):
            if img[y][x] != 255: count += 1
            if count > width * 0.10: continue
        if count > width * 0.10 or count == 0: 
            # print(count, width * 0.10)
            continue
        else: 
            pass
            # print("MENOR", count)
    
        for x in range(0, width, 3):
            count = 0
            for px in range(x, width):
                if img[y][px] != 255:
                    count += 1
                else: 
                    break
            
            if count < 3:
                for px in range(x, x+2):
                    if px >= width: break
                    img[y][px] = 255
                if show_progress:
                    mostra_pontos(img, [Ponto(x,y)], delay)
                
            
        





def ajusta_angulo(img):
    pass

def ajusta_imagem(img, show_progress = False, delay = 0):
    img = satura(img, show_progress, delay)
    remove_pixel_isolado(img, show_progress, delay)
    remove_bordas(img, show_progress, delay)
    pass

def start():
    path, dirs, files = next(os.walk("./bases"))
    print(path, dirs)

    dir_index = -1
    while dir_index < 0 or dir_index >= len(dirs):
        system("cls")
        print("Escolha Quais imagens trabalhar")
        for dir_name in dirs:
            print(dir_name)
        dir_index = int(input()) - 1
        pass
    
    dir_name = dirs[dir_index] + "/"
    bases = carrega_img_base(dir_name)
    defeitos = carrega_img_def(dir_name)
    img_base = bases[0]
    ajusta_imagem(img_base, True, 3)


start()