import re

def getIdOfProduct(p_link):
    link_temp = re.split(r"[?/]",p_link)
    for l in link_temp:
        if l.find("B")==0:
            if len(l)==10:
                return l
    return "Not Found"

def isFake(star, foundStar):
    if(abs(foundStar - star) >= 2):
        return True
    else:
        return False

def findStar(polarity):
    if(polarity < -0.6):
        return 1
    elif(polarity < -0.2):
        return 2
    elif(polarity > -0.2 and polarity < 0.2):
        return 3
    elif(polarity > 0.2 and polarity < 0.6):
        return 4
    else:
        return 5
