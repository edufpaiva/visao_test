'''
    - roteiro seguido: https://www.pucsp.br/~jarakaki/pai/Roteiro4.pdf
    - link rotacao: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    - python -m pip install --upgrade pip 
    - python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
    - pip install opencv-python
    - separei em funcoes, pq acho que assim da pra entender melhor oq cada coisa faz
        e nao ter que ficar memorisando funcao da biblioteca

'''
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

def merge_sort(vet):
    if len(vet) == 1:
        return vet
    mid = int(len(vet)/2)
    return merge(merge_sort(vet[:mid]), merge_sort(vet[mid:]))

def merge(vet_1, vet_2):
    i = j = 0
    ret = []
    while i < len(vet_1) and j < len(vet_2):
        pass
        if vet_1[i] < vet_2[j]:
            ret.append(vet_1[i])
            i += 1
        else:
            ret.append(vet_2[j])
            j += 1

    for k in range(i, len(vet_1)):
        ret.append(vet_1[k])
    for k in range(j, len(vet_2)):
        ret.append(vet_2[k])
    
    return ret

#   Le uma imagem
def read_img(path):
    #AlunosPUC_Oracle.jpg')#, cv2.IMREAD_GRAYSCALE )
    # Le a imagem para uma matriz
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

#   Mostra img na tela
def show_resized_img(img, percent, name='image'):
    dim = img.shape
    print(img.shape)
    
    dim = (int(dim[0] * percent / 100), int(dim[1] * percent / 100))
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    name='image'    #   Comment if u wanto to rename windows

    cv2.imshow(name, resized)
    print("============================\n\tpress enter\n============================\n")
    cv2.waitKey(0)

#   Mostra img redimensionada na tela
def show_img(img, name='image'):
    result = img
    
    name='image'    #   Comment if u wanto to rename windows

    cv2.imshow(name, cv2.resize(result, (1024, 640), interpolation= cv2.INTER_AREA))
    print("============================\n\tpress enter\n============================\n")
    cv2.waitKey(0)

#   Print em console: widht, height and channels of img
def print_img_params(img):
    print(img.shape)
    print ("width: {} pixels".format(img.shape[1]))
    print ("height: {} pixels".format(img.shape[0]))
    if len(img.shape) >= 3: print ("channels: {}".format(img.shape[2])) #   bites por pixel

def read_and_show_img(path):
    print("img: ", path)
    img = read_img(path)
    show_img(img)
    print_img_params(img)
    print()

def get_img_base(path = "./"):
    imagens = []
    for i in range(1,N_IMAGENS_BASE+1):
        imagens.append(read_img( path +"base (%s).jpg" % i))
    return imagens

def get_img_def(path = "./"):
    imagens = []
    for i in range(1,N_IMAGENS_DEF + 1):
        imagens.append(read_img(path +"def (%s).jpg" % i))
    return imagens


#   Media de pixels da imagem - para imagens BGR 
def media(vet):
    total = 0
    for i in vet:
        total += i
    return int(total/len(vet))

def percorre_pixels(img):
    # print(img.shape)
    # print_img_params(img)
    # height, width, channels = img.shape # comment if use in grayscale
    
    height, width = img.shape
    video = cv2.VideoWriter('ult.avi',cv2.VideoWriter_fourcc(*'DIVX'),15, (width, height))
    
    for py in range(0, height):
        for px in range(0,  width):
            # pixel = int(media(img[py][px]) / 50) * 50 #   comment is use iin grayscale
            pixel = int(img[py][px] / 50) * 50
            if pixel >= 200: pixel = 255
            if pixel <= 100 : pixel = 0
            img[py][px] = pixel
            video.write(img)

            # for pixel_index in range(len(img[py][px])): #   comment is use iin grayscale
            #     img[py][px][pixel_index] = pixel

    video.release()

    return img
    # show_img(img)

def get_empty_img(img):
    h_1, w_1 = img.shape[:2]
    return np.zeros((h_1, w_1, 3), np.uint8)

def get_empty_img_grayscale(img):
    h_1, w_1 = img.shape[:2]
    return np.zeros((h_1, w_1, 1), np.uint8)

def compare_img(img1, img2):
    print("comparando")
    h_1, w_1 = img1.shape[:2]
    h_2, w_2 = img2.shape[:2]
    # print(h_1, h_2, w_1, w_2)

    result = np.zeros((h_1, w_1, 3), np.uint8)
    count = 0
    for py in range(h_1):
        if py % 30 == 0: 
            # cv2.imshow("image", cv2.resize(result, (1024, 640), interpolation= cv2.INTER_AREA))
            show_img(result)
        # if py % 10 == 0: 
        #     count += 1
        #     cv2.imwrite("Resultado/result%s.jpg" %(count), result)
        for px in range(w_1):
            try:
                if img1[py][px] == 255 and img2[py][px] == 255: continue
                if compare_pixel(img1, img2, py, px):
                    result[py][px][1] = 255
                    pass
                else: 
                    result[py][px][2] = 255
            except:
                # print(py, px)
                result[py][px][0] = 255
                pass
    # show_img(result)
    # show_resized_img(result, RESIZE)
    return result
    
def compare_pixel(img1, img2, y, x):
    # if img1[y][x] == img2[y][x]: return True 
    # else: return False

    pixel = img1[y][x]
    if pixel == 255: return False

    h_1, w_1 = img1.shape
    
    for i in range(y+1, y+2):
        if i > img2.shape[0]: continue
        for j in range(x+1, x+2):
            if i >= h_1: continue
            if j >= w_1: continue
            if i < 0 or j < 0: continue
            if img2[i][j] != 255: return True
    return False

def get_ponto_superior(img):
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            if img[y][x] != 255:
                return Ponto(x, y)

    return Ponto(0, 0)

def get_pontos_linha_superior(img, ponto, step=10):
    h, w = img.shape

    pontos = []

    if ponto.x < w/2:
        for x in range(ponto.x, w, step):
            for y in range(int(h/4)):
                if img[y][x] != 255:
                    pontos.append(Ponto(x,y))
                    break
    else:
        for x in range(w, ponto.x, -1 *step):
            for y in range(int(h/4)):
                try:
                    pass
                    if img[y][x] != 255:
                        pontos.append(Ponto(x,y))
                        break
                except :

                    pass
    return pontos

def print_ponto(img, ponto, bgr = [0, 0, 255]):
    # print(img.shape)

    h_1, w_1, channel = img.shape
    x = ponto["x"]
    y = ponto["y"]

    for i in range(y-2, y+4):
        for j in range(x-2, x+4):
            img[i][j] = bgr
    return img

def mostra_ponto(img, ponto):

    result = get_empty_img(img)
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            result[y][x] = [img[y][x], img[y][x], img[y][x]]

    for y in range(ponto.y-2, ponto.y + 3):
        for x in range(ponto.x-2, ponto.x + 3):
            result[y][x] = [0, 0, 255]
    show_img(result, "PONTO")

def get_angulo(p1, p2):
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

def format_img(img):
    
    ponto_mais_alto = get_ponto_superior(img)
    ponto_mais_a_esqueda = get_ponto_esquerda(img)    
    pontos_superiores = get_pontos_linha_superior(img, ponto_mais_alto, 1)
    angulos = []
    
    # print("ponto mais alto", ponto_mais_alto.to_string())
    # print(pontos_superiores)
    
    for ponto in pontos_superiores:
        angulos.append(get_angulo(ponto_mais_alto, ponto))
    # print(angulos)
    moda = get_moda(angulos)
    # print("moda:", moda)
    img = rotate_bound(img, -moda)
    # show_img(img)
    cv2.imwrite("02 - Rotacionada.jpg", img)


    img = retira_bordas(img)  
    
    return img

def retira_bordas(img):
    h, w = img.shape[:2]
    print(w,h)
    for y in range(h-1, 0, -1):
        for x in range(w-1, 0, -1):
            if img[y][x] == 255: break
            img[y][x] = 255
    for y in range(h):
        for x in range(w):
            if img[y][x] == 255: break
            img[y][x] = 255
    for x in range(w-1, 0, -1):
        for y in range(h-1, 0, -1):
            if img[y][x] == 255: break
            img[y][x] = 255
    for x in range(w):
        for y in range(h):
            if img[y][x] == 255: break
            img[y][x] = 255
    
    return img

def get_moda(vet):
    vet_len = len(vet)
    if vet_len%2 == 0:
        vet_len = int(vet_len/2)
        return (vet[vet_len] + vet[vet_len-1])/2
    else:
        return vet[int(vet_len/2)]

def get_media(vet):
    total = 0
    for i in vet: total += i
    return total / len(vet)

def get_ponto_esquerda(img):
    h, w = img.shape[:2]
    pontos = []
    for y in range(int(h/4)):
        for x in range(int(w/4)):
            if img[y][x] != 255:
                pontos.append(x)
                break
    pontos = merge_sort(pontos)
    print("PONTOS")
    # print(pontos)
    moda = int(get_moda(pontos))
    print(Ponto(moda, 0).to_string())
    return Ponto(moda, 0)

def alinha_imagem_com_a_outra(img1, img2):

    h, w = img2.shape[:2]

    x1 = get_ponto_esquerda(img1).x
    x2 = get_ponto_esquerda(img2).x

    y1 = get_ponto_superior(img1).y
    y2 = get_ponto_superior(img2).y

    off_y = y2 - y1
    off_x = x2 - x1

    print(off_x)

    result = get_empty_img_grayscale(img2)

    result[:h][:w] = 255

    for y in range(h):
        for x in range(w):
            if img2[y][x] != 255: 
                try:
                    pass
                    result[y-off_y][x-off_x] = img2[y][x]
                except:
                    print(y-off_y, y, x-off_x, x)
                    pass
            
                

    result = retira_bordas(result)
    return result




    return result

def ajuste_erros(img):
    pass
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            if y >= h: continue
            if x >= w: continue
            try:
                if img[y][x][2] == 255:
                    img[y][x] = verifica_pixel_vermelho_unico(img, y, x)
            except :
                print("width: %s,\t height: %s\nx: %s,\t y: %s" %(w,h,x,y))
                input()
    return img

def verifica_pixel_vermelho_unico(img, y, x):
    h, w = img.shape[:2]
    count = 0
    for py in range(y-2, y+3):
        for px in range(x-2, x+3):
            if py >= h: continue
            if px >= w: continue
            if py == y and px == x: continue
            # if img[py][px][2] == 255: return [0,0,255]
            if img[py][px][2] == 255: count += 1

    if count >= 3: return [0,0,255]
    return [255,0,0]

def start():
    bases =  get_img_base(PATH_BASE)
    defeitos = get_img_def(PATH_DEF)

    h, w = defeitos[INDEX_DEF].shape[:2]

    #   Mostra imagem antes de qualquer alteracao
    show_img(defeitos[INDEX_DEF], "IMAGEM SEM ALTERACOES")

    print("SATURANDO IMG DEF")
    img2 = percorre_pixels(defeitos[INDEX_DEF])
    cv2.imwrite("01 - Saturacao.jpg", img2)


    #   Mostra Imagem com pixels saturados
    print("IMG DEF SATURADA")
    show_img(img2, "IMAGEM SATURADA")

    print("SATURANDO IMG BASE")
    img1 = percorre_pixels(bases[INDEX_BASE])

    img1 = format_img(img1)
    img2 = format_img(img2)

    #   Mostra imagem angulada Corrigida
    print("IMAGEM ANGULO CORRIGIDO")
    show_img(img2, "ANGULO CORRIGIDO")
    cv2.imwrite("03 - Angulada Sem borda.jpg", img2)



    img2 = alinha_imagem_com_a_outra(img1, img2)


    #   Mostra Imagem Alinhada com a imagem base 
    print("POSICIONAMENTO CORRIGIDO")
    show_img(img2, "POSICAO CORRIGIDA")
    cv2.imwrite("04 - Alinhada.jpg", img2)


    print("COMPARANDO IMAGEM")
    result = compare_img(img1, img2)

    #   Mostra o Resultado da comparacao, a imagem original utilizada para fazer a mascara, e a imagem comparada
    print("DEFININDO ERROS")
    result = ajuste_erros(result)

    cv2.imwrite("result.jpg", result)

    print("RESULTADO")
    show_img(result, "RESULTADO COMPARACAO")
    show_img(img1, "BASE")
    show_img(img2, "IMG TESTADA")

    #   Mostra o resultado Menor para uma visão geral
    show_resized_img(result, RESIZE)


# r = 63
# bases =  get_img_base(PATH_BASE)
# show_img(bases[INDEX_BASE])
# show_img(np.uint8(bases[INDEX_BASE]/r) * r)
# show_img(bases[INDEX_BASE])
# show_img(percorre_pixels(bases[INDEX_BASE]))
# show_img(np.uint8(bases[INDEX_BASE]/r) * r)
start()


