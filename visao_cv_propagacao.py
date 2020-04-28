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
import subprocess

system("cls")

import cv2
import numpy as np
import math

import tkinter as tk
from tkinter import messagebox

from datetime import datetime

PATH_BASE = "./bases/01 - completo/"
PATH_DEF = "./defeitos/01 - completo/"
N_IMAGENS_BASE = 1
N_IMAGENS_DEF = 3
INDEX_BASE = 0
INDEX_DEF = 1
RESIZE = 30
# RESIZE = 100
BLUE    = [255,  0 ,  0 ]
GREEN   = [150, 255, 150]
RED     = [ 0 ,  0 , 255]
BLACK   = [ 0 ,  0 ,  0 ]
PINK    = [219, 203, 255]
WHITE   = [255, 255, 255]
YELLOW  = [ 0 , 215, 255]

PRINT_RESULT = False
PRINT_COUNT = 1

PATH = "Resultado-%02d-%02d-%04d-%02d%02d%02d" % (datetime.now().day, datetime.now().month, datetime.now().year, datetime.now().hour, datetime.now().minute, datetime.now().second)

class Ponto:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_string(self):
        return("x: %i, y: %i" %(self.x, self.y))

def print_result(img):
    
    global PRINT_COUNT
    cv2.imwrite("Print/Print-%s.png" %(PRINT_COUNT), img)
    PRINT_COUNT += 1

def get_angulo( p2, p1):
    if p1.x-p2.x == 0: return 0
    return math.degrees(math.atan((p1.y-p2.y)/(p1.x-p2.x)))

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def get_moda(vet):
    vet.sort()
    print(vet)
    vet_len = len(vet)
    if vet_len == 0: return 0
    if vet_len%2 == 0:
        vet_len = int(vet_len/2)
        return (vet[vet_len] + vet[vet_len-1])/2
    else:
        return vet[int(vet_len/2)]

def get_empty_img(img, height=None, width=None):
    h_1, w_1 = img.shape[:2]
    if height != None and width != None:
        h_1, w_1 = height, width
    return np.zeros((h_1, w_1, 3), np.uint8)

def get_empty_img_grayscale(img, height=None, width=None):
    h_1, w_1 = img.shape[:2]
    if height != None and width != None:
        h_1, w_1 = height, width
    return np.zeros((h_1, w_1, 1), np.uint8)

def copia_colorida(img):
    try:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    except:
        h, w = img.shape[:2]
        copy = get_empty_img(img)
        for y in range(h):
            for x in range(w):
                copy[y][x] = img[y][x]
        return copy

def copia_escala_cinza(img):
    try:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        h, w = img.shape[:2]
        copy = get_empty_img_grayscale(img)
        for y in range(h):
            for x in range(w):
                copy[y][x][0] = img[y][x]
        return copy

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

def zoom_img(img, py, px, precision = 50, delay=0, name="zoom", width=400, height=400):
    zoom = get_empty_img(img, precision*2, precision*2)
    h, w = img.shape[:2]
    py -= precision
    px -= precision

    for y in range(precision*2):
        for x in range(precision * 2):
            if py + y <  0: continue
            if py + y >= h: continue
            if px + x <  0: continue
            if px + x >= w: continue
            zoom[y][x] = img[py + y][px + x]

    show_img(zoom, name, delay, height, width)
    return zoom

def show_img(img, name='image', delay = 0, height=640, width=1024):
    result = img
    
    # name='image'    #   Comment if u wanto to rename windows

    cv2.imshow(name, cv2.resize(result, (width, height), interpolation= cv2.INTER_AREA))
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

def circula_pontos(img, pontos, delay=0, tamanho = 4, expessura = 1):
    height, width = img.shape[:2]
    color =  copia_colorida(img)

    for ponto in pontos:
        x1 = ponto.x - tamanho
        x2 = ponto.x + tamanho + 1
        y1 = ponto.y - tamanho
        y2 = ponto.y + tamanho + 1
        if x1 < 0: x1 = 0
        if x2 > width: x2 = width
        if y1 < 0: y1 = 0
        if y2 > height: y2 = height

        for n in range(expessura):

            for x in range(x1+n, x2-n):
                color[y1+n][x] = [0, 0, 255]
                color[y2-n][x] = [0, 0, 255]
            for y in range(y1+n, y2-n):
                color[y][x1+n] = [0, 0, 255]
                color[y][x2-n] = [0, 0, 255]
        

    # show_img(color, "progress", delay)
    return color

def linha_vertical(img, ponto, show_progress = False, delay = 0):
    height, width = img.shape[:2]
    color = copia_colorida(img)

    for y in range(height):
        color[y][ponto.x] = [0, 255, 0]
        if show_progress and y % int(height/10) == 0: show_img(color, "progress", delay)
    
    show_img(color, "progress", 3)
    if PRINT_RESULT:print_result(color)

    return color

def linha_horizontal(img, ponto, show_progress = False, delay = 0):
    height, width = img.shape[:2]
    color = copia_colorida(img)

    for x in range(width):
        color[ponto.y][x] = [255, 0, 0]
        if show_progress and x % int(width/10) == 0: show_img(color, "progress", delay)
    
    show_img(color, "progress", 3)

    if PRINT_RESULT:print_result(color)

    return color

def get_pixel_mais_acima(img):
    height, width = img.shape[:2]

    off_cima = 0
    for y in range(height):
        for x in range(width):
            if img[y][x] != 255: return Ponto(x, y)

def get_pixel_mais_abaixo(img):
    height, width = img.shape[:2]

    off_cima = 0
    for y in range(height-1, 0, -1):
        for x in range(width):
            if img[y][x] != 255: return Ponto(x, y)

def get_pixel_mais_a_esquerda(img):
    height, width = img.shape[:2]

    off_cima = 0
    for x in range(width):
        for y in range(height):
            if img[y][x] != 255: return Ponto(x, y)

def get_pixel_mais_a_direita(img):
    height, width = img.shape[:2]

    off_cima = 0
    for x in range(width-1, 0, -1):
        for y in range(height):
            if img[y][x] != 255: return Ponto(x, y)

def remove_bordas(img, show_progress = False, delay = 0):
    height, width = img.shape[:2]

    off_cima = get_pixel_mais_acima(img)
    off_esquerda = get_pixel_mais_a_esquerda(img)
    
    if show_progress:
        print(off_cima.to_string())
        print(off_esquerda.to_string())
        color = mostra_pontos(img, [off_cima, off_esquerda])
        color = linha_vertical(color, off_esquerda, show_progress, delay)
        color = linha_horizontal(color, off_cima, show_progress, delay)
        color = mostra_pontos(color, [Ponto(off_esquerda.x, off_cima.y)])

    copy = get_empty_img_grayscale(img)
    copy[:][:] = 255

    for y in range(height):
        py = y + off_cima.y
        if py >= height: break
        if show_progress and y % int(height / 20) == 0:
            show_img(copy, "progress", delay)
        for x in range(width):
            px = x + off_esquerda.x
            if px >= width: break

            copy[y][x] = img[py][px]
    
    if show_progress: show_img(copy, "progress")
    img = copy

    off_direita = get_pixel_mais_a_direita(img)
    off_baixo = get_pixel_mais_abaixo(img)
    
    if show_progress:
        color = mostra_pontos(img, [off_direita, off_baixo])
        color = linha_vertical(color, off_direita, show_progress, delay)
        color = linha_horizontal(color, off_baixo, show_progress, delay)
        color = mostra_pontos(color, [Ponto(off_direita.x, off_baixo.y)])

    copy = get_empty_img(img, off_baixo.y, off_direita.x)

    h, w = copy.shape[:2]

    for x in range(w):
        for y in range(h):
            copy[y][x] = img[y][x]
    
    if show_progress: show_img(copy, "progress")
    
    try:
        return cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY)
    except:
        print("ERROR AJUSTA IMAGEM CONVERTER PARA GRAY")
        return copy
    
def satura(img, show_progress = False, delay = 0):
    height, width = img.shape[:2]
    
    if show_progress: 
        show_img(img, "progress")
        print("SATURANDO IMAGEM")


    for py in range(0, height):
        

        if py % int(height/10) == 0 and show_progress: 
            linha_horizontal(img, Ponto(0, py), show_progress, delay)

        for px in range(0,  width):
            pixel = int(img[py][px] / 50) * 50
            if pixel >= 200: pixel = 255
            if pixel <= 50 : pixel = 0
            img[py][px] = pixel
    
    if show_progress:
        show_img(img, "progress", delay)

    r = 63
    img = np.uint8(img/r) * r

    if PRINT_RESULT:print_result(img)

    
    not_colored = []
    for py in range(0, height):

        if show_progress and py % int(height/10) == 0: 
            linha_horizontal(img, Ponto(0, py), show_progress, delay)

        for px in range(0,  width):
            if img[py][px] >= 200:  img[py][px] = 255
            elif img[py][px] <= 50:  img[py][px] = 0
            elif img[py][px] > 50 and img[py][px] <= 130: img[py][px] = 100
            elif img[py][px] > 130 and img[py][px] < 200: img[py][px] = 150
            elif img[py][px] not in not_colored: not_colored.append(img[py][px])


    if show_progress: 
        print("IMAGEM SATURADA")
        show_img(img, "progress")

    if PRINT_RESULT:print_result(img)

    return img
                        
def verifica_relevancia_do_pixel(img, ponto, show_progress, delay):
    tam = 50
    height, width = img.shape[:2]
    if show_progress:
        zoom = get_empty_img(img, tam, tam)
        for y in range(tam):
            for x in range(tam):
                py = y + ponto.y - int(tam/2)
                px = x + ponto.x - int(tam/2)
                if py >= height or py < 0: continue
                if px >= width or px < 0: continue
                zoom[y][x] = img[py][px]

        show_img(circula_pontos(img,  [ponto], delay)  , 'progress', delay)
        show_img(circula_pontos(zoom, [Ponto(int(tam/2), int(tam/2))], delay), 'zoom', delay, tam*6, tam*6)

    x = ponto.x
    y = ponto.y

    # 0 0 0
    # 0 1 0
    # 0 0 0
    if img[y-1][x-1] == 255 and img[y-1][ x ] == 255 and img[y-1][x+1] == 255 and img[ y ][x-1] == 255 and img[ y ][x+1] == 255 and img[y+1][x-1] == 255 and img[y+1][ x ] == 255 and img[y+1][x+1] == 255: return True

    # 1 0 0
    # 1 P 0
    # 1 0 0
    
    if img[y-1][x-1] != 255 and img[y-1][ x ] == 255 and img[y-1][x+1] == 255 and img[ y ][x-1] != 255 and img[ y ][x+1] == 255 and img[y+1][x-1] != 255 and img[y+1][ x ] == 255 and img[y+1][x+1] == 255: return True

    # 1 1 1
    # 0 P 0
    # 0 0 0
    if img[y-1][x-1] != 255 and img[y-1][ x ] != 255 and img[y-1][x+1] != 255 and img[ y ][x-1] == 255 and img[ y ][x+1] == 255 and img[y+1][x-1] == 255 and img[y+1][ x ] == 255 and img[y+1][x+1] == 255: return True

    # 0 0 1
    # 0 P 1
    # 0 0 1
    if img[y-1][x-1] == 255 and img[y-1][ x ] == 255 and img[y-1][x+1] != 255 and img[ y ][x-1] == 255 and img[ y ][x+1] != 255 and img[y+1][x-1] == 255 and img[y+1][ x ] == 255 and img[y+1][x+1] != 255: return True

    # 0 0 0
    # 0 P 0
    # 1 1 1
    if img[y-1][x-1] == 255 and img[y-1][ x ] == 255 and img[y-1][x+1] == 255 and img[ y ][x-1] == 255 and img[ y ][x+1] == 255 and img[y+1][x-1] != 255 and img[y+1][ x ] != 255 and img[y+1][x+1] != 255: return True

    # 0 0 0 0
    # 0 P 1 0
    # 0 0 0 0
    if img[y-1][x-1] == 255 and img[y-1][ x ] == 255 and img[y-1][x+1] == 255 and img[y-1][x+2] == 255 and img[ y ][x-1] == 255 and img[ y ][x+1] != 255 and img[ y ][x+2] == 255 and img[y+1][x-1] == 255 and img[y+1][ x ] == 255 and img[y+1][x+1] == 255 and img[y+1][x+2] == 255: return True

    # 0 0 0 0
    # 0 P 0 0
    # 0 1 0 0
    # 0 0 0 0
    if img[y-1][x-1] == 255 and img[y-1][ x ] == 255 and img[y-1][x+1] == 255 and img[ y ][x-1] == 255 and img[ y ][x+1] == 255 and img[y+1][x-1] == 255 and img[y+1][ x ] != 255 and img[y+1][x+1] == 255 and img[y+2][x-1] == 255 and img[y+2][ x ] == 255 and img[y+2][x+1] == 255: return True
    
    # 0 0 0 0
    # 0 P 0 0
    # 0 0 1 0
    # 0 0 0 0
    if img[y-1][x-1] == 255 and img[y-1][ x ] == 255 and img[y-1][x+1] == 255 and img[y-1][x+2] == 255 and img[ y ][x-1] == 255 and img[ y ][x+1] == 255 and img[ y ][x+2] == 255 and img[y+1][x-1] == 255 and img[y+1][ x ] == 255 and img[y+1][x+1] != 255 and img[y+1][x+2] == 255 and img[y+2][x-1] == 255 and img[y+2][ x ] == 255 and img[y+2][x+1] == 255 and img[y+2][x+2] == 255: return True

    #   0 0 0
    # 0 0 P 0
    # 0 1 0 0
    # 0 0 0 
    if img[y-1][x-1] == 255    and img[y-1][ x ] == 255    and img[y-1][x+1] == 255 and img[ y ][x-2] == 255    and img[ y ][x-1] == 255                                and img[ y ][x+1] == 255 and img[y+1][x-2] == 255    and img[y+1][x-1] != 255    and img[y+1][ x ] == 255    and img[y+1][x+1] == 255 and img[y+2][x-2] == 255    and img[y+2][x-1] == 255    and img[y+2][ x ] == 255                            : return True

    # 0 0 1 0
    # 0 P 1 0
    # 0 1 1 0
    # 0 0 1 0
    if img[y-1][x-1] == 255 and img[y-1][ x ] == 255 and img[y-1][x+1] != 255 and img[ y ][x-1] == 255 and img[ y ][x+1] != 255 and img[y+1][x-1] == 255 and img[y+1][ x ] != 255 and img[y+1][x+1] != 255 and img[y+2][x-1] == 255 and img[y+2][ x ] == 255 and img[y+2][x+1] != 255: return True

    
    # 1 0 0
    # 1 P 0
    # 1 1 0
    # 1 0 0
    if img[y-1][x-1] != 255 and img[y-1][ x ] == 255 and img[y-1][x+1] == 255 and img[ y ][x-1] != 255 and img[ y ][x+1] == 255 and img[y+1][x-1] != 255 and img[y+1][ x ] != 255 and img[y+1][x+1] == 255 and img[y+2][x-1] != 255 and img[y+2][ x ] == 255 and img[y+2][x+1] == 255: return True

    # 1 1 1 1
    # 0 P 1 0
    # 0 0 0 0
    if img[y-1][x-1] != 255 and img[y-1][ x ] != 255 and img[y-1][x+1] != 255 and img[y-1][x+2] != 255 and img[ y ][x-1] == 255 and img[ y ][x+1] != 255 and img[ y ][x+2] == 255 and img[y+1][x-1] == 255 and img[y+1][ x ] == 255 and img[y+1][x+1] == 255 and img[y+1][x+2] == 255 and img[y+2][x-1] == 255 and img[y+2][ x ] == 255 and img[y+2][x+1] == 255 and img[y+2][x+2] == 255: return True

    # 0 0 0 0
    # 0 P 1 0
    # 1 1 1 1
    if img[y-1][x-1] == 255 and img[y-1][ x ] == 255 and img[y-1][x+1] == 255 and img[y-1][x+2] == 255 and img[ y ][x-1] == 255 and img[ y ][x+1] != 255 and img[ y ][x+2] == 255 and img[y+1][x-1] != 255 and img[y+1][ x ] != 255 and img[y+1][x+1] != 255 and img[y+1][x+2] != 255: return True

    # 0 0 0 0 0
    # 0 0 0 0 0
    # 0 0 P 0 0
    # 0 0 0 0 0
    # 0 0 0 0 0
    # if  
    #     img[y-2][x-2] == 255 and img[y-2][x-1] == 255 and img[y-2][ x ] == 255 and img[y-2][x+1] == 255 and img[y-2][x+2] == 255
    # and img[y-1][x-2] == 255 and img[y-1][x-1] == 255 and img[y-1][ x ] == 255 and img[y-1][x+1] == 255 and img[y-1][x+2] == 255
    # and img[ y ][x-2] == 255 and img[ y ][x-1] == 255                          and img[ y ][x+1] == 255 and img[ y ][x+2] == 255
    # and img[y+1][x-2] == 255 and img[y+1][x-1] == 255 and img[y+1][ x ] == 255 and img[y+1][x+1] == 255 and img[y+1][x+2] == 255
    # and img[y+2][x-2] == 255 and img[y+2][x-1] == 255 and img[y+2][ x ] == 255 and img[y+2][x+1] == 255 and img[y+2][x+2] == 255: return True

    return False
    pass

def remove_pixel_isolado(img, show_progress = False, delay = 0):
    n_img = img.copy()
    h, w = img.shape[:2]

    delay_erro = delay
    show_line = False

    # for y in range(h):
    #     if show_progress:
    #         if y % int(h/10) == 0: 
    #             linha_horizontal(n_img, Ponto(0, y), show_progress, delay)
    #     for x in range(w):
    #         if n_img[y][x] < 255: n_img[y][x] = 0

    if show_progress:
        root = tk.Tk()
        S = tk.Scrollbar(root)
        T = tk.Text(root, height=4, width=50)
        S.pack(   side=tk.RIGHT, fill=tk.Y)
        T.pack(   side=tk.LEFT, fill=tk.Y)
        S.config( command=T.yview)
        T.config( yscrollcommand=S.set)
        quote = "O processo de limpeza da imagem é demorado."
        T.insert(tk.END, quote)

        msg = tk.messagebox.askquestion("Visualizar", "Visualizar processo?", icon="warning")
        if msg == 'yes': 
            msg2 = tk.messagebox.askquestion("Visualizar", "Pausar em cada erro encontrado??", icon="warning")
            if msg2 == 'yes':
                delay_erro = 0
            else:
                delay_erro = 1
            root.destroy()
        else: 
            show_progress = False
            show_line = True
            root.destroy()
        
        root.mainloop()

    for x in range(1, w -1):
        
        if show_line  and x % int(w/20) == 0: linha_vertical(img, Ponto(x,0), False, 3)
        
        for y in range(1, h -1):
            try:
                if img[y-1][x] == 255 and img[y][x] != 255 and img[y+1][x] == 255:
                    if verifica_relevancia_do_pixel(img, Ponto(x, y), show_progress, delay):
                        img[y][x] = 255
                        if show_progress: show_img(circula_pontos(img, [Ponto(x, y)]), "progress", delay_erro)
            except:
                pass
            try:    
                if img[y][x-1] == 255 and img[y][x] != 255 and img[y][x+1] == 255:
                    if verifica_relevancia_do_pixel(img, Ponto(x, y), show_progress, delay):
                        img[y][x] = 255
                        if show_progress: show_img(circula_pontos(img, [Ponto(x, y)]), "progress", delay_erro)
            except:
                pass
            try:
                if img[y-1][x] == 255 and img[y][x] != 255 and img[y+1][x] != 255 and img[y+2][x] == 255:
                    if verifica_relevancia_do_pixel(img, Ponto(x, y), show_progress, delay):
                        img[y][x] = 255
                        if show_progress: show_img(circula_pontos(img, [Ponto(x, y)]), "progress", delay_erro)
            except:
                pass
            try:
                if img[y][x-1] == 255 and img[y][x] != 255 and img[y][x+1] != 255 and img[y][x+2] == 255:
                    if verifica_relevancia_do_pixel(img, Ponto(x, y), show_progress, delay):
                        img[y][x] = 255
                        if show_progress: show_img(circula_pontos(img, [Ponto(x, y)]), "progress", delay_erro)
            except:
                pass
                    
def ajusta_angulo(img, show_progress = False, delay = 0):
    h, w = img.shape[:2]
    pixel_branco = False
    for y in range(h):
        if img[y][0] == 255:
            pixel_branco = True
            break

    if pixel_branco:
        pontos = []

        ponto_mais_alto = get_pixel_mais_acima(img)
        if w < 300:

            for x in range(ponto_mais_alto.x, w, 50):
                for y in range(50):
                    if img[y][x] != 255:
                        pontos.append(Ponto(x,y))
                        break
        else:
            for x in range(0, w, int(w/30)):
                for y in range(50):
                    if img[y][x] != 255:
                        pontos.append(Ponto(x,y))
                        break
        mostra_pontos(img, pontos, 0)
        angulos = []
        for i in range( len(pontos)):
            angulo = -get_angulo(ponto_mais_alto, pontos[i])
            # print(angulo)
            if angulo > 3 or angulo < -3: continue
            angulos.append(angulo)
        
        copy = get_empty_img_grayscale(img, h+100, w+100)
        copy[:][:] = 255
        for y in range(h):
            for x in range(w):
                copy[y+50][x+50] = img[y][x]

        angulo = get_moda(angulos)
        print(angulo)
        if angulo > 2 and angulo < 3 or angulo > -3 and angulo < -2:
            angulo = angulo/2
        else:
            return img
        copy = rotate_bound(copy, angulo)
        show_img(copy, "progress")

        h, w = copy.shape[:2]

        for y in range(h):
            for x in range(w):
                if copy[y][x] == 255: break
                else: copy[y][x] = 255
        for x in range(w):
            for y in range(h):
                if copy[y][x] == 255: break
                else: copy[y][x] = 255
        for y in range(h-1, 0, -1):
            for x in range(w-1, 0, -1):
                if copy[y][x] == 255: break
                else: copy[y][x] = 255
        for x in range(w-1, 0, -1):
            for y in range(h-1, 0, -1):
                if copy[y][x] == 255: break
                else: copy[y][x] = 255

        show_img(copy, "progress")

        copy = remove_bordas(copy, show_progress, delay)

        show_img(copy,"progress")

        return copy
    else: return img

def verifica_pixel_valido(img1, img2, py, px):
    h, w = img1.shape[:2]

    for y in range(py-1, py+2):
        for x in range(px-1, px+2):
            if y < 0 or x < 0 or y >= h or x >= w: continue
            if y == py and x == px:continue
            if img1[y][x] == img2[py][px]:
                return True
    return False
    pass

def propaga(img1, img2, result, y, x):

    h, w = img1.shape[:2]

    try:
        if verif_cor_pixel(result[y][x], RED): 
            print("VERMELHO")
            return 0
        if verif_cor_pixel(result[y][x], BLUE): 
            print("AZUL")
            return 0
        if img1[y][x] == 255 and img2[y][x] == 255: return 0            
        elif img1[y][x] < 255 and img2[y][x] < 255: return 0
        else: 
            if verifica_pixel_valido(img1, img2, y, x): return 0
            else: 
                result[y][x] = BLUE
                direita, esquerda, cima, baixo = 0, 0, 0, 0
                if x + 1 < w: direita = propaga(img1, img2, result, y, x + 1)
                if x - 1 > 0: esquerda = propaga(img1, img2, result, y, x - 1)
                if y - 1 > 0: cima = propaga(img1, img2, result, y-1, x)
                if y + 1 < h: baixo = propaga(img1, img2, result, y+1, x)
                return 1 + direita + esquerda + cima + baixo
    except:
        return 0

def pinta_pixel_proximos(img, y, x, color):
    h, w = img.shape[:2]

    if verif_cor_pixel(img[y][x], BLUE): 
        img[y][x] = color
        if x + 1 < w: direita   = pinta_pixel_proximos(img, y, x + 1, RED)
        if x - 1 > 0: esquerda  = pinta_pixel_proximos(img, y, x - 1, RED)
        if y - 1 > 0: cima      = pinta_pixel_proximos(img, y - 1, x, RED)
        if y + 1 < h: baixo     = pinta_pixel_proximos(img, y + 1, x, RED)
    
def verifica_linha(img, y, x):
    hor = 0
    ver = 0
    
    h, w = img.shape[:2]

    try:
        if not verif_cor_pixel(img[y][x-1], BLUE) and verif_cor_pixel(img[y][x], BLUE) and not verif_cor_pixel(img[y][x+1], BLUE)  : ver +=1
        if not verif_cor_pixel(img[y][x-1], BLUE) and verif_cor_pixel(img[y+1][x], BLUE) and not verif_cor_pixel(img[y][x+1], BLUE): ver +=1
        if not verif_cor_pixel(img[y][x-1], BLUE) and verif_cor_pixel(img[y+2][x], BLUE) and not verif_cor_pixel(img[y][x+1], BLUE): ver +=1
        if not verif_cor_pixel(img[y][x-1], BLUE) and verif_cor_pixel(img[y+3][x], BLUE) and not verif_cor_pixel(img[y][x+1], BLUE): ver +=1
        if not verif_cor_pixel(img[y][x-1], BLUE) and verif_cor_pixel(img[y+4][x], BLUE) and not verif_cor_pixel(img[y][x+1], BLUE): ver +=1
    except:
        pass

    if ver >= 3: return True

    try:
        if not verif_cor_pixel(img[y-1][x], BLUE) and verif_cor_pixel(img[y][x], BLUE) and not verif_cor_pixel(img[y+1][x], BLUE)  : hor +=1
        if not verif_cor_pixel(img[y-1][x], BLUE) and verif_cor_pixel(img[y][x+1], BLUE) and not verif_cor_pixel(img[y+1][x], BLUE): hor +=1
        if not verif_cor_pixel(img[y-1][x], BLUE) and verif_cor_pixel(img[y][x+2], BLUE) and not verif_cor_pixel(img[y+1][x], BLUE): hor +=1
        if not verif_cor_pixel(img[y-1][x], BLUE) and verif_cor_pixel(img[y][x+3], BLUE) and not verif_cor_pixel(img[y+1][x], BLUE): hor +=1
        if not verif_cor_pixel(img[y-1][x], BLUE) and verif_cor_pixel(img[y][x+4], BLUE) and not verif_cor_pixel(img[y+1][x], BLUE): hor +=1
    except:
        pass
    
    if hor >= 3: return True

    return False

def compara_img(img1, img2, show_progress, delay):
    h,  w  = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h), interpolation= cv2.INTER_AREA)

    pontos = []
    # result = copia_colorida(img1)
    result = get_empty_img(img1)

    for y in range(h):
        if y % int(h/20) == 0 and show_progress: 
            linha_horizontal(result, Ponto(0, y), show_progress, delay)
            if PRINT_RESULT:print_result(result)
        
        for x in range(w):
            
            try:
                if img1[y][x] == 255 and img2[y][x] == 255: continue            
                elif img1[y][x] < 255 and img2[y][x] < 255: result[y][x] = GREEN
                else: 
                    if verifica_pixel_valido(img1, img2, y, x): result[y][x] = BLACK
                    else:
                        if verif_cor_pixel(result[y][x], RED): continue
                        error = propaga(img1, img2, result, y, x)
                        if error > 3:
                            if not verifica_linha(result, y, x):
                                pinta_pixel_proximos(result, y, x, RED)
                                if show_progress: show_img(result, "progress", 1)
                                pontos.append(Ponto(x, y))
            except:
                result[y][x] = WHITE

    if PRINT_RESULT:print_result(result)

    show_img(result,'progress')
    # cv2.imwrite("Result2.png", result)
    cv2.imwrite("%s/Comparacao.png" %(PATH), result)

    return {"pontos":pontos, "result":result}

def ajusta_imagem(img, show_progress = False, delay = 0):
    img = satura(img, show_progress, delay)
    remove_pixel_isolado(img, show_progress, delay)
    # img = remove_bordas(img, show_progress, delay)
    # img = ajusta_angulo(img, show_progress, delay)
    show_img(img, "progress", delay)
    
    try:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except:
        print("ERROR AJUSTA IMAGEM CONVERTER PARA GRAY")
        return img

def verif_cor_pixel(pixel, color):
    for i in range(len(pixel)):
        if pixel[i] != color[i]: return False
    return True

def checa_validadae_erro(img, py, px):
    #   pixel de erro 2x2
    tam_pixel_erro = 2
    
    for y in range(py, py+tam_pixel_erro):
        for x in range(px, px+tam_pixel_erro):
            if not verif_cor_pixel(img[y][x], BLUE): return False
    return True

def limpa_falso_positivo(img, show_progress, delay):
    """
    @return a list of points that possibly have errors

    """
    
    pontos = []
    h, w = img.shape[:2]
    for y in range(h):
        if show_progress and y % 50 == 0: 
            show_img(img, "progress", delay)
            if PRINT_RESULT:print_result(img)
        for x in range(w):
            if verif_cor_pixel(img[y][x], BLUE):
                if checa_validadae_erro(img, y, x):
                    pontos.append(Ponto(x,y))
                    for py in range(y, y+3):
                        for px in range(x, x+3):
                            img[py][px] = RED
                    if show_progress: 
                        show_img(img, "progress", delay)
    if show_progress: show_img(img, "progress")
    if PRINT_RESULT:print_result(img)
    cv2.imwrite("Result_Erros.png", img)
    return pontos

def remove_pontos_proximos( pontos, delete_range=40):
    # print(len(pontos))
    if len(pontos) == 0: return pontos
    result = []
    
    
    pontos.reverse()
    result.append(pontos.pop())

    while len(pontos) > 0:
        ponto = pontos.pop()
        
        y = result[-1].y
        x = result[-1].x
        
        if ponto.x > x and ponto.x < x + delete_range or ponto.y > y and ponto.y < y + delete_range: continue

        result.append(ponto)
    # print(len(result))
    return result

def pergunta_yes_no(titulo, texto):
    msg = tk.messagebox.askquestion(titulo, texto, icon="warning")
    if msg == 'yes': 
        return True
    else: 
        return False
        
    
    # root.mainloop()


def junta_tres_imagens(img1, img2, img3):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h3, w3 = img3.shape[:2]

    result = get_empty_img(img1, h1, w1+w2+w3)

    img1 = copia_colorida(img1)
    img2 = copia_colorida(img2)
    img3 = copia_colorida(img3)

    for y in range(h1):
        for x in range(w1):
            
            result[y][x] = img1[y][x]

    for y in range(h2):
        for x in range(w2):
            
            result[y][x + w1] = img2[y][x]
    for y in range(h3):
        for x in range(w3):
            result[y][x + w1 + w2] = img3[y][x]
            
    return result


def start(index_base = 0, index_def = 0):

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
    bases    = carrega_img_base(dir_name)
    defeitos = carrega_img_def(dir_name)
    # img_base = bases[1]
    # img_def  = defeitos[2]

    # img_base = bases[index_base]
    # img_def  = defeitos[index_def]

    img_base = copia_escala_cinza(bases[index_base])
    img_def = copia_escala_cinza(defeitos[index_def])


    # show_img(junta_tres_imagens(zoom_img(img_base, 1000, 1100), zoom_img(img_def, 1100, 1100), zoom_img(img_base, 900, 900)))


    # img_def  = defeitos[6]
    # img_def  = defeitos[3]
    img_def  = ajusta_imagem(img_def, True, 3)
    print("IMAGEM AJUSTADA")
    img_base = ajusta_imagem(img_base, True, 3)
    print("IMAGEM AJUSTADA")
    
    comparacao = compara_img(img_base, img_def, True, 3)
    
    erros = comparacao["pontos"]
    result = comparacao["result"]


    print("COMPARACAO FEITA")
    # erros = limpa_falso_positivo(result, True, 3)        
    print("ERROS DETECTADOS")

    # erros = remove_pontos_proximos(erros)

    erros_confirmados = []

    root = tk.Tk()
    S = tk.Scrollbar(root)
    T = tk.Text(root, height=4, width=50)
    S.pack(   side=tk.RIGHT, fill=tk.Y)
    T.pack(   side=tk.LEFT, fill=tk.Y)
    S.config( command=T.yview)
    T.config( yscrollcommand=S.set)
    quote = "Verificando Similaridade de Imagens."
    T.insert(tk.END, quote)
    for ponto in erros:

        z1 = zoom_img(bases[index_base], ponto.y, ponto.x, delay=1, name="Original")
        z2 = zoom_img(defeitos[index_def] , ponto.y, ponto.x, delay=1, name="Comparativo")
        z3 = zoom_img(result , ponto.y, ponto.x, delay=1, name="Localizacao")
        
        zz = junta_tres_imagens(z1,z2,z3)
        cv2.imwrite("%s/erro_%s_%s.png" %(PATH, ponto.x, ponto.y), zz)

        # show_img(zz, 'ZOOM MASTER')
        if PRINT_RESULT: print_result(zz)

        if pergunta_yes_no("São iguais", "Tem erros na imagem?"): 
            erros_confirmados.append(ponto)

    root.mainloop()

    result = circula_pontos(img_def, erros_confirmados, tamanho=60, expessura=5)

    show_img(result, "progress", 0)
    cv2.imwrite("%s/Resultado_final.png" %(PATH), result)

system('mkdir ' + PATH)
start()
subprocess.Popen('explorer "%s"' %(PATH))
