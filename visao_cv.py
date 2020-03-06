'''
    - roteiro seguido: https://www.pucsp.br/~jarakaki/pai/Roteiro4.pdf
    - separei em funcoes, pq acho que assim da pra entender melhor oq cada coisa faz
        e nao ter que ficar memorisando funcao da biblioteca

'''
from os import system
system("cls")

import cv2
import numpy as np

#   Le uma imagem
def read_img(path):
    #AlunosPUC_Oracle.jpg')#, cv2.IMREAD_GRAYSCALE )
    # Le a imagem para uma matriz
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

#   Mostra img na tela
def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

#   Print em console: widht, height and channels of img
def print_img_params(img):
    print(img.shape
    )
    print ("width: {} pixels".format(img.shape[1]))
    print ("height: {} pixels".format(img.shape[0]))
    print ("channels: {}".format(img.shape[2])) #   bites por pixel


def read_and_show_img(path):
    print("img: ", path)
    img = read_img(path)
    show_img(img)
    print_img_params(img)
    print()

def get_img_base():
    imagens = []
    for i in range(1,7):
        imagens.append(read_img("base%s.jpg" % i))
    return imagens

def mostra_imagens_base():
    #   mostra todas a imagens contidas na pasta
    #   que se chamam base e sao numeradas de 1 a 6 ;D hehe
    for i in range(1,7):
        read_and_show_img("base%s.jpg" % i)

def media(vet):
    total = 0
    for i in vet:
        total += i
    return int(total/len(vet))

def percorre_pixels(img):
    print(img.shape)
    # height, width, channels = img.shape # comment if use in grayscale
    height, width = img.shape
    for py in range(0, height):
        for px in range(0,  width):
            # pixel = int(media(img[py][px]) / 50) * 50 #   comment is use iin grayscale
            pixel = int(img[py][px] / 50) * 50
            if pixel >= 200: pixel = 255
            if pixel <= 100 : pixel = 0
            img[py][px] = pixel

            # for pixel_index in range(len(img[py][px])): #   comment is use iin grayscale
            #     img[py][px][pixel_index] = pixel

    show_img(img)


# mostra_imagens_base()

imagens =  get_img_base()
percorre_pixels(imagens[0])







