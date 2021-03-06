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

# IF TRUE, GENERATES IMAGES FROM VARIOUS PARTS OF THE IMAGE TREATMENT PROCESS
PRINT_RESULT = False
PRINT_COUNT = 1

PATH_T0_SAVE_IMG = "Resultado-%02d-%02d-%04d-%02d%02d%02d" % (datetime.now().day, datetime.now().month, datetime.now().year, datetime.now().hour, datetime.now().minute, datetime.now().second)

class Ponto:
    """
        @class - Representa um ponto no plano cartesiano
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_string(self):
        """
            Imprime os valores do ponto
        """
        return("x: %i, y: %i" %(self.x, self.y))


def print_result(img:img, name:str="Print", extension:str="png", path:str="Print/"):
    """
        Salva uma imagem.\n
        @param img: cv2 img\n
            \tImagem para ser salva.\n
        @param name : str\n
            \tNome do arquivo.\n
        @param extension : str\n
            \tExtensao do arquivo.\n
        @param path : str\n
            \tEndereco do diretorio em que o arquivo deve ser salvo.\n
    """
    global PRINT_COUNT
    cv2.imwrite("%s%s-%s.%s" %(path, name, PRINT_COUNT, extension), img)
    PRINT_COUNT += 1

def get_angulo( p1:Ponto, p2:Ponto)->float:
    """
        Calcula o angulo entre dois pontos.\n
        @param p1: Ponto\n
            \tPonto 1\n
        @param p2: Ponto\n
            \tPonto 2\n
    """
    if p2.x-p1.x == 0: return 0
    return math.degrees(math.atan((p2.y-p1.y)/(p2.x-p1.x)))

def rotate_bound(image, angle:float)->img:
    """
    Rotaciona uma imagem sem perda de conteudo.\n
    @param image: cv2 image\n
        \tA imagem a ser rotacionada.\n
    @param angle: float\n
        \tO angulo em graus que a imagem devera ser rotacionada.

    
    """

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

def get_moda(vet:list)->float:
    """
        Calcula a moda de uma vetor\n
        @param vet:list\n
            \tVetor para ser retirado a moda.\n
    """
    vet.sort()
    print(vet)
    vet_len = len(vet)
    if vet_len == 0: return 0
    if vet_len%2 == 0:
        vet_len = int(vet_len/2)
        return (vet[vet_len] + vet[vet_len-1])/2
    else:
        return vet[int(vet_len/2)]

def get_empty_img(height:int=400, width:int=400, grayscale:bool=False)->img:
    """
    Cria uma imagem vazia\n
    @param height : int\n
        \tA altura da imagem.\n
    @param width : int\n
        \tA largura da imagem.\n
    @param grayscale: bool\n
        \tSe verdadeiro a imagem retornada sera em escala de cinza
    """
    if grayscale: return np.zeros((height, width, 1), np.uint8)
    else: return np.zeros((height, width, 3), np.uint8)

def copia_colorida(img:img)->int:
    """
    Copia uma imagem e converte para colorida.\n
    @param img: cv2 image.\n
        \tImagem para ser copiada e convertida\n
    """
    try:
        return cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    except:
        return img.copy()

def copia_escala_cinza(img:img)->int:
    """
    Copia uma imagem e converte para escala de cinza.\n
    @param img: cv2 image.\n
        \tImagem para ser copiada e convertida\n
    """
    try:
        return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    except:
        return img.copy()

def carrega_img_base(dir_name:str)->list:
    """
    Carrega imagen base para a memoria.\n
    @param dir_name : str\n
        \tO nome do dirertorio onde estao as imagens.\n
    """
    path, dirs, files = next(os.walk("./bases/" + dir_name))
    
    imagens = []
    for file_name in files:
        imagens.append(cv2.imread(path + file_name, cv2.IMREAD_GRAYSCALE))
    return imagens

def carrega_img_def(dir_name:str)->list:
    """
    Carrega imagen de defeito para a memoria.\n
    @param dir_name : str\n
        \tO nome do dirertorio onde estao as imagens.\n
    """
    path, dirs, files = next(os.walk("./defeitos/" + dir_name))
    imagens = []
    for file_name in files:
        imagens.append(cv2.imread(path + file_name, cv2.IMREAD_GRAYSCALE))
    return imagens

def zoom_img(img:img, ponto:Ponto, precision:int = 50)->int:
    """
        Da um zoom em determinado ponto da imagem.\n
        @param img: cv2 img\n
            \tA imagem base para ser dado o zoom\n
        @param ponto: Ponto\n
            \tPonto onde sera dado o zoom
        @param precision: int\n
            \tA quantidade de pixels na imagem gerada\n
    """
    zoom = get_empty_img(img, precision*2, precision*2)
    h, w = img.shape[:2]
    
    py, px = ponto.y, ponto.x

    py -= precision
    px -= precision

    for y in range(precision*2):
        for x in range(precision * 2):
            if py + y <  0: continue
            if py + y >= h: continue
            if px + x <  0: continue
            if px + x >= w: continue
            zoom[y][x] = img[py + y][px + x]

    return zoom
    """
        Da um zoom em determinado ponto da imagem.\n
        @param img: cv2 img\n
            \tA imagem base para ser dado o zoom\n
        @param py: int\n
            \tA coordenada y do pixel\n
        @param px: int\n
            \tA coordenada x do pixel\n
        @param precision: int\n
            \tA quantidade de pixels na imagem gerada\n
    """
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

    return zoom

def show_img(img:img, name:str='image', delay:int = 0, height:int=640, width:int=1024):
    """
        Exibe uma imagem em tela.\n
        @param img: cv2 img\n
            \tImagem a ser exibida na tela.\n
        @param name: str\n
            \tTitulo da imagem.\n
        @param delay: int\n
            \tTempo que a imagem sera exibida na tela em milissegundos.\n
            \tSe o valor for '0' aguardara que o usuario pressione uma tecla.\n
        @param height: int\n
            \tAltura da imagem.\n
        @param width: int\n
            \tLargura da imagem.\n
    """
    result = img
    
    # name='image'    #   Comment if u wanto to rename windows

    cv2.imshow(name, cv2.resize(result, (width, height), interpolation= cv2.INTER_AREA))
    if delay == 0: print("============================\n\tpress enter\n============================\n")
    cv2.waitKey(delay)

def contorna_pontos(img:img, pontos:list, distancia:int = 2, expessura:int = 2, color:list=RED)->int:
    """
        Contorna pontos na imagem.\n
        @param img: cv2 image\n
            \tImagem em que o pontos serao contornados.\n
        @param pontos: Ponto[]\n
            \tPontos a serem circulados na imagem.\n
        @param distancia: int\n
            \tDistancia do contorno ao ponto central em pixels.\n
        @param expessura: int\n
            \tExpessura do contorno em pixels.\n
        @param color: int[3] | int[4].\n
            \tCor do contorno\n
    """
    height, width = img.shape[:2]
    color =  copia_colorida(img)

    for ponto in pontos:
        x1 = ponto.x - distancia
        x2 = ponto.x + distancia + 1
        y1 = ponto.y - distancia
        y2 = ponto.y + distancia + 1
        if x1 < 0: x1 = 0
        if x2 > width: x2 = width
        if y1 < 0: y1 = 0
        if y2 > height: y2 = height

        for n in range(expessura):

            for x in range(x1+n, x2-n):
                color[y1+n][x] = color
                color[y2-n][x] = color
            for y in range(y1+n, y2-n):
                color[y][x1+n] = color
                color[y][x2-n] = color

    return color

def linha_vertical(img:img, ponto:Ponto, show_progress:bool = False, delay:int = 0, color:list=GREEN)->int:
    """
        Cria uma linha vertical na imagem.\n
        @param img:\n
            \tImagem onde sera tracada a linha.\n
        @param ponto:Ponto\n
            \tPonto base para a linha ser tracada.\n
        @param show_progress:bool\n
            \tSe verdadeiro mostra a linha sendo tracada na imagem.\n
        @param delay:int\n
            \tO tempo que cada imagem ficara na tela em milissegundos.\n
        @param color:list\n
            \tA cor da linha.\n
    """
    height, width = img.shape[:2]
    img_color = copia_colorida(img)

    for y in range(height):
        img_color[y][ponto.x] = color
        if show_progress and y % int(height/10) == 0: show_img(img_color, "progress", delay)
    
    if show_progress: show_img(img_color, "progress", delay)
    if PRINT_RESULT:print_result(img_color)

    return img_color

def linha_horizontal(img:img, ponto:Ponto, show_progress:bool = False, delay:int = 0, color:list=YELLOW)->int:
    """
        Cria uma linha horizontal na imagem.\n
        @param img:\n
            \tImagem onde sera tracada a linha.\n
        @param ponto:Ponto\n
            \tPonto base para a linha ser tracada.\n
        @param show_progress:bool\n
            \tSe verdadeiro mostra a linha sendo tracada na imagem.\n
        @param delay:int\n
            \tO tempo que cada imagem ficara na tela em milissegundos.\n
        @param color:list\n
            \tA cor da linha.\n
    """
    height, width = img.shape[:2]
    img_color = copia_colorida(img)

    for x in range(width):
        img_color[ponto.y][x] = color
        if show_progress and x % int(width/10) == 0: show_img(img_color, "progress", delay)
    
    # show_img(color, "progress")

    if PRINT_RESULT:print_result(img_color)

    return img_color

def get_pixel_mais_acima(img:img)->Ponto:
    """
        Enconta o pixel valido mais alto da imagem.\n
        @param img: cv2 img\n
            \tImagem para encontrar o ponto.\n
    """
    height, width = img.shape[:2]

    off_cima = 0
    for y in range(height):
        for x in range(width):
            if img[y][x] != 255: return Ponto(x, y)

def get_pixel_mais_abaixo(img:img)->Ponto:
    """
        Enconta o pixel valido mais abaixo da imagem.\n
        @param img: cv2 img\n
            \tImagem para encontrar o ponto.\n
    """
    height, width = img.shape[:2]

    off_cima = 0
    for y in range(height-1, 0, -1):
        for x in range(width):
            if img[y][x] != 255: return Ponto(x, y)

def get_pixel_mais_a_esquerda(img:img)->Ponto:
    """
        Enconta o pixel valido mais a esquerda da imagem.\n
        @param img: cv2 img\n
            \tImagem para encontrar o ponto.\n
    """
    height, width = img.shape[:2]

    off_cima = 0
    for x in range(width):
        for y in range(height):
            if img[y][x] != 255: return Ponto(x, y)

def get_pixel_mais_a_direita(img:img)->Ponto:
    """
        Enconta o pixel valido mais a direita da imagem.\n
        @param img: cv2 img\n
            \tImagem para encontrar o ponto.\n
    """
    height, width = img.shape[:2]

    off_cima = 0
    for x in range(width-1, 0, -1):
        for y in range(height):
            if img[y][x] != 255: return Ponto(x, y)

def remove_bordas(img:img, show_progress:bool = False, delay:int = 0)->int:
    """
        Remove os espacos em branco ao redor da imagem.\n
        @param img: cv2 img\n
            \tImagem para ser removido as bordas.\n
        @param show_progress:bool\n
            \tSe verdadeiro mostra o processo de remocao das bordas.\n
        @param delay:int\n
            \tTempo que cada imagem aparece na tela em milissegundos\n
    """

    height, width = img.shape[:2]

    off_cima = get_pixel_mais_acima(img)
    off_esquerda = get_pixel_mais_a_esquerda(img)
    
    if show_progress:
        print(off_cima.to_string())
        print(off_esquerda.to_string())
        color = contorna_pontos(img, [off_cima, off_esquerda])
        color = linha_vertical(color, off_esquerda, show_progress, delay)
        color = linha_horizontal(color, off_cima, show_progress, delay)
        color = contorna_pontos(color, [Ponto(off_esquerda.x, off_cima.y)])

    copy = get_empty_img(height, width, True)
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
        color = contorna_pontos(img, [off_direita, off_baixo])
        color = linha_vertical(color, off_direita, show_progress, delay)
        color = linha_horizontal(color, off_baixo, show_progress, delay)
        color = contorna_pontos(color, [Ponto(off_direita.x, off_baixo.y)])

    copy = get_empty_img(off_baixo.y, off_direita.x)

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
    
def satura_img(img:img, show_progress:bool = False, delay:int = 0)->int:
    """
        Ajusta as cores da imagem para especificos tons de preto cinza e branco
        tornando-os padroes. \n
        (0, 100, 150, 255).\n
        @param img:img\n
            \tImagem a ser saturada.\n
        @param show_progress:bool\n
            \tSe verdadeiro mostra o processo de saturacao da imagem.\n
        @param delay:int\n
            \tTempo em que cada imagem sera exibida na tela em milissegundos.\n
    """


    height, width = img.shape[:2]
    
    if show_progress: 
        show_img(img, "progress", delay)
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
                        
def verifica_relevancia_do_pixel(img:img, ponto:Ponto, show_progress:bool=False, delay:int=1)->bool:
    """
        Verifica se o pixel e relevante para a composicao da imagem.\n
        @param img:img\n
            \tImagem onde o pixem se encontra\n
        @param pixel:Ponto\n
            \tCoordenada do pixel.\n
        @apram show_progress:bool\n
            \tSe verdadeiro exibe o pixel.\n
        @param delay:int\n
            \tTempo em que cada imagem e exibida na tela\n
    """
    tam = 50
    height, width = img.shape[:2]
    if show_progress:
        zoom = zoom_img(img, ponto, 50)

        show_img(contorna_pontos(img,  [ponto], delay)  , 'progress', delay)
        show_img(contorna_pontos(zoom, [Ponto(int(tam/2), int(tam/2))], delay), 'zoom', delay, tam*6, tam*6)

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

def remove_pixel_isolado(img:img, show_progress:bool = False, delay:int = 0)->None:
    """
        Remove pixels de sujeira da imagem.\n
        @param img:cv2 img\n
            \tA imagem a ser limpa\n
        @apram show_progress:bool\n
            \tSe verdadeiro exibe o processo.\n
        @param delay:int\n
            \tTempo em que cada imagem e exibida na tela\n
    """
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
                    if show_progress: show_img(contorna_pontos(img, [Ponto(x, y)]), "progress", delay_erro)
                    
            if img[y][x-1] == 255 and img[y][x] != 255 and img[y][x+1] == 255:
                if verifica_relevancia_do_pixel(img, Ponto(x, y), show_progress, delay):
                    img[y][x] = 255
                    if show_progress: show_img(contorna_pontos(img, [Ponto(x, y)]), "progress", delay_erro)
                    
            if img[y-1][x] == 255 and img[y][x] != 255 and img[y+1][x] != 255 and img[y+2][x] == 255:
                if verifica_relevancia_do_pixel(img, Ponto(x, y), show_progress, delay):
                    img[y][x] = 255
                    if show_progress: show_img(contorna_pontos(img, [Ponto(x, y)]), "progress", delay_erro)

            if img[y][x-1] == 255 and img[y][x] != 255 and img[y][x+1] != 255 and img[y][x+2] == 255:
                if verifica_relevancia_do_pixel(img, Ponto(x, y), show_progress, delay):
                    img[y][x] = 255
                    if show_progress: show_img(contorna_pontos(img, [Ponto(x, y)]), "progress", delay_erro)
                    
def ajusta_angulo(img:img, show_progress:bool = False, delay:int = 0)->int:
    
    """
        Rotaciona a imagem se ela estiver angulada para um dos lados.\n
        @param img: cv2 img\n
            \tImagem a ser angulada\n
        @param show_progress:bool\n
            \tSe verdadeiro exibe o processo.\n
        @param delay:int\n
            \tTempo em que cada imagem e exibida na tela\n
    """

    h, w = img.shape[:2]
    
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
    
    contorna_pontos(img, pontos)
    angulos = []
    
    for i in range( len(pontos)):
        angulo = -get_angulo(ponto_mais_alto, pontos[i])
        # print(angulo)
        if angulo > 3 or angulo < -3: continue
        angulos.append(angulo)
    
    copy = get_empty_img(h+100, w+100, True)
    copy[:][:] = 255
    
    for y in range(h):
        for x in range(w):
            copy[y+50][x+50] = img[y][x]

    angulo = get_moda(angulos)
    # print(angulo)
    
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

def verifica_pixel_valido(img1:img, img2:img, py:int, px:int)->bool:
    """
        Verifica se há algum pixel semelhante proximo ao pixel selecionado.\n
        @param img1: cv2 img\n
            \tImagem de base para comparacao.\n
        @param img2: cv2 img\n
            \tImagem para verificar se os pixels semelhantes.\n
        @param py: int\n
            \tponto y de coordenada do pixel.\n
        @param px: int\n
    """        
    h, w = img1.shape[:2]

    for y in range(py-1, py+2):
        for x in range(px-1, px+2):
            if y < 0 or x < 0 or y >= h or x >= w: continue
            if y == py and x == px:continue
            if img1[y][x] == img2[py][px]:
                return True
    return False
    pass

def compara_img(img1, img2, show_progress:bool, delay:int)->int:
    """
        Compara duas imagens para gerar uma mascara de resultado.\n
        @param img1: cv2 img\n
            \tImagem base para comparacao.\n
        @param img2: cv2 img\n
            \tImagem a ser comparada com a base para achar possiveis falhas.\n
        @param show_progress: bool\n
            \tSe verdadeiro mostra o prcesso de verificacao.\n
        @param delay: int\n
            \tTempo que cada imagem aparecera na tela, em milissegundos.\n
    """  
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
                        result[y][x] = BLUE
                        pontos.append(Ponto(x, y))
            except:
                result[y][x] = WHITE

    if PRINT_RESULT:print_result(result)

    if show_progress: show_img(result,'progress', delay)
    cv2.imwrite("Result2.png", result)
    return result

def ajusta_imagem(img, show_progress:bool = False, delay:int = 0)->int:
    """
        Ajusta imagem para verificacao de erros.\n
        (satura, limpa, angula, remove bordas)\n
        @param img: cv2 img\n
            \tImagem para ser ajustada.\n
        @param show_progress:bool\n
            \tSe verdadeiro mostra o o processo de ajuste da imagem.\n
        @param delay:int\n
            \tTempo em que cada imagem sera exibida na tela. \n
            \tEm milissegundos.\n
    """

    img = satura_img(img, show_progress, delay)
    # remove_pixel_isolado(img, show_progress, delay)
    # img = remove_bordas(img, show_progress, delay)
    # img = ajusta_angulo(img, show_progress, delay)
    show_img(img, "progress", delay)
    
    try:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except:
        print("ERROR AJUSTA IMAGEM CONVERTER PARA GRAY")
        return img

def verif_cor_pixel(pixel:list, color:list)->bool:
    """
        Verifica a cor do pixel e a mesma que a cor passada.\n
        @param pixel: list\n
            \tPixel para verificar a cor.\n
        @param color: list\n
            \tCor base para verificacao do pixel.\n
    """
    
    for i in range(len(pixel)):
        if pixel[i] != color[i]: return False
    return True

def checa_validadae_erro(img, py:int, px:int)->bool:
    """
        Verifica se o tamanho do erro é grande o suficiente para ser um erro valido, 
        como um quadrado 2x2.\n
        @param img: cv2 img\n
            \tA imagem para ser validado o erro.\n
        @param py: int\n
            \tCoodernada y do ponto inicial do erro.\n
        @param px: int\n
            \tCoodernada x do ponto inicial do erro.\n
    """

    #   pixel de erro 2x2
    tam_pixel_erro = 2
    
    for y in range(py, py+tam_pixel_erro):
        for x in range(px, px+tam_pixel_erro):
            if not verif_cor_pixel(img[y][x], BLUE): return False
    return True

def limpa_falso_positivo(img, show_progress:bool, delay:int)->list:
    """
    Verifica os erros encontrados na imagem e exclui aqueles que possivelmente nao são erros validos\n

    @param img: cv2 img\n
        \tA imagem de mascara para verificar os erros encontrados.\n
    @param  show_progress: bool\n
        \tSe verdadeiro exibe o processo de limpeza da imagem\n
    @param  delay: int\n
        \tO tempo em que cada imagem sera exibida na tela.\n
        \tEm milissegundos.\n

    @return a list of points that possibly have errors\n

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

def remove_pontos_proximos(pontos:list, delete_range:int=40)->list:
    """
        Remove pontos selecionados proximos um ao outro dentro de um range, 
        para evitar que o mesmo trecho da imagem seja exibido mais de uma vez.\n
        @param pontos: list\n
            \tLista de pontos selecionados para verificação.\n
        @paaram delete_range: int\n
            \tRange que sera removido os pontos que estiverem dentro\n
    """
    # print(len(pontos))
    if len(pontos) == 0: return pontos
    pontos_selecionados = []
    
    
    pontos.reverse()
    pontos_selecionados.append(pontos.pop())

    while len(pontos) > 0:
        ponto = pontos.pop()
        
        y = pontos_selecionados[-1].y
        x = pontos_selecionados[-1].x
        
        if ponto.x > x and ponto.x < x + delete_range or ponto.y > y and ponto.y < y + delete_range: continue

        pontos_selecionados.append(ponto)
    # print(len(result))
    return pontos_selecionados

def pergunta_yes_no(titulo:str, texto:str)->None:
    """
        Utiliza o Tkinter para exibir uma mensagem de confimação na tela.\n
        @param titulo: str\n
            \tTitulo da mensagem\n
        @param texto: str\n
            \t Conteudo da mensagem\n
    """

    msg = tk.messagebox.askquestion(titulo, texto, icon="warning")
    if msg == 'yes': 
        return True
    else: 
        return False
        
    
    # root.mainloop()

def junta_tres_imagens(img1, img2, img3)->int:
    """
        Une 3 imagens de mesmo tamanho em uma unica imagem.\n
        img1, img2, img3: cv2 img\n
            \tImagem para uniao.\n
    """

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
    
    result = compara_img(img_base, img_def, True, 3)
    print("COMPARACAO FEITA")
    erros = limpa_falso_positivo(result, True, 3)        
    print("ERROS DETECTADOS")

    erros = remove_pontos_proximos(erros)

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
        # show_img(zz, 'ZOOM MASTER')
        if PRINT_RESULT: print_result(zz)

        if pergunta_yes_no("São iguais", "Tem erros na imagem?"): 
            erros_confirmados.append(ponto)


    root.mainloop()

    result = contorna_pontos(img_def, erros_confirmados, tamanho=60, expessura=5)

    show_img(result, "progress", 0)
    system('mkdir ' + PATH_T0_SAVE_IMG)
    cv2.imwrite("%s/Resultado_final.png" %(PATH_T0_SAVE_IMG), result)


start()
subprocess.Popen('explorer "%s"' %(PATH_T0_SAVE_IMG))

