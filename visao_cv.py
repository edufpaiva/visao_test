'''
    - roteiro seguido: https://www.pucsp.br/~jarakaki/pai/Roteiro4.pdf
    - separei em funcoes, pq acho que assim da pra entender melhor oq cada coisa faz
        e nao ter que ficar memorisando funcao da biblioteca

'''


import cv2

#   Le uma imagem
def read_img(path):
    #AlunosPUC_Oracle.jpg')#, cv2.IMREAD_GRAYSCALE )
    img = cv2.imread(path)
    return img

#   Mostra img na tela
def show_img(img):
    cv2.imshow('image', read_img(path))



def read_and_show_img(path):
    img = read_img(path)
    show_img(img)
    cv2.waitKey(0)

def mostra_imagens_base():
    #   mostra todas a imagens contidas na pasta
    #   que se chamam base e sao numeradas de 1 a 6 ;D hehe
    for i in range(1,7):
        read_and_show_img("base%s.jpg" % i)



mostra_imagens_base()
