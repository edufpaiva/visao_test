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

import tkinter as tk
from tkinter import messagebox

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

def circula_pontos(img, pontos, delay=0):
    height, width = img.shape[:2]
    color =  copia_colorida(img)

    for ponto in pontos:
        x1 = ponto.x - 3
        x2 = ponto.x + 4
        y1 = ponto.y - 3
        y2 = ponto.y + 4
        if x1 < 0: x1 = 0
        if x2 > width: x2 = width
        if y1 < 0: y1 = 0
        if y2 > height: y2 = height
        for x in range(x1, x2):
            color[y1][x] = [0, 0, 255]
            color[y2][x] = [0, 0, 255]
        for y in range(y1, y2):
            color[y][x1] = [0, 0, 255]
            color[y][x2] = [0, 0, 255]


    # show_img(color, "progress", delay)
    return color

def linha_vertical(img, ponto, show_progress = False, delay = 0):
    height, width = img.shape[:2]
    color = copia_colorida(img)

    for y in range(height):
        color[y][ponto.x] = [0, 255, 0]
        if show_progress and y % int(height/10) == 0: show_img(color, "progress", delay)
    
    # show_img(color, "progress")
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
        if show_progress:
            if py % int(height/10) == 0: 
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
            if py % int(height/10) == 0: 
                linha_horizontal(img, Ponto(0, py), show_progress, delay)
                # show_img(img, "progress", delay)
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
        
        if show_line  and x % int(w/20) == 0: linha_vertical(img, Ponto(x,0), show_line, delay)
        
        for y in range(1, h -1):
            if img[y-1][x] == 255 and img[y][x] != 255 and img[y+1][x] == 255:
                if verifica_relevancia_do_pixel(img, Ponto(x, y), show_progress, delay):
                    img[y][x] = 255
                    if show_progress: show_img(circula_pontos(img, [Ponto(x, y)]), "progress", delay_erro)
                    
            if img[y][x-1] == 255 and img[y][x] != 255 and img[y][x+1] == 255:
                if verifica_relevancia_do_pixel(img, Ponto(x, y), show_progress, delay):
                    img[y][x] = 255
                    if show_progress: show_img(circula_pontos(img, [Ponto(x, y)]), "progress", delay_erro)
                    
            if img[y-1][x] == 255 and img[y][x] != 255 and img[y+1][x] != 255 and img[y+2][x] == 255:
                if verifica_relevancia_do_pixel(img, Ponto(x, y), show_progress, delay):
                    img[y][x] = 255
                    if show_progress: show_img(circula_pontos(img, [Ponto(x, y)]), "progress", delay_erro)

            if img[y][x-1] == 255 and img[y][x] != 255 and img[y][x+1] != 255 and img[y][x+2] == 255:
                if verifica_relevancia_do_pixel(img, Ponto(x, y), show_progress, delay):
                    img[y][x] = 255
                    if show_progress: show_img(circula_pontos(img, [Ponto(x, y)]), "progress", delay_erro)
                    
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


def compara_img(img1, img2, show_progress, delay):
    h,  w  = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h), interpolation= cv2.INTER_AREA)

    # result = copia_colorida(img1)
    result = get_empty_img(img1)

    for y in range(h):
        if y % int(h/30) == 0: linha_horizontal(result, Ponto(0, y), show_progress, delay)
        for x in range(w):
            
            try:
                if img1[y][x] == 255 and img2[y][x] == 255: continue            
                elif img1[y][x] < 255 and img2[y][x] < 255: result[y][x] = [0,255,0]
                else: 
                    
                    if verifica_pixel_valido(img1, img2, y, x): result[y][x] = [0,0,0]
                    else: result[y][x] = [0,0,255]
            except:
                result[y][x] = [255,0,0]
                
            
    show_img(result,'progress')
    cv2.imwrite("Result2.png", result)



def ajusta_imagem(img, show_progress = False, delay = 0):
    img = satura(img, show_progress, delay)
    # remove_pixel_isolado(img, show_progress, delay)
    # img = remove_bordas(img, show_progress, delay)
    # img = ajusta_angulo(img, show_progress, delay)
    show_img(img, "progress", delay)
    
    try:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except:
        print("ERROR AJUSTA IMAGEM CONVERTER PARA GRAY")
        return img

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
    bases    = carrega_img_base(dir_name)
    defeitos = carrega_img_def(dir_name)
    img_base = bases[1]
    img_def  = defeitos[3]
    # img_def  = defeitos[6]
    # img_def  = defeitos[3]
    img_def  = ajusta_imagem(img_def, True, 3)
    img_base = ajusta_imagem(img_base, True, 3)
    
    compara_img(img_base, img_def, True, 3)
    
start()