from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.models import User
from .models import Product, Review
from .forms import ProductForm
from . import summarize, functions
import heartrate; heartrate.trace(browser=True)

# Scraping
import itertools
import requests
import math
import time
import random
import string
import re
from bs4 import BeautifulSoup
from textblob import TextBlob

text_str = ""

review_count = 0
fake_count = 0
review_star_sum = 0

def about(request):
    return render(request, 'analyzer/about.html',{'title':'About'})

def home(request):
    if request.method == 'POST':
        url_entered = request.POST['p_url']
        start_time = time.time()

        url = "https://www.amazon.com/product-reviews/"+functions.getIdOfProduct(url_entered)+"/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber="
        # url = "https://www.amazon.com/product-reviews/B07GLV1VC7/ref=cm_cr_getr_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber="

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',} # To handle OVERLOAD
        r = requests.post(url+"1", headers=headers)
        r.raise_for_status()

        soup = BeautifulSoup(r.content,'lxml') # Parse content
        #-----------------------------------------------------------------------------------
        # Functions
        #-----------------------------------------------------------------------------------

        def getReviewCount():
            r_number = soup.findAll("span",attrs={"data-hook":"cr-filter-info-review-count"})
            r_count = r_number[0].text.split(" ")[3]
            return int(r_count)

        def getProductName():
            p_name = soup.findAll("a",attrs={"data-hook":"product-link"})
            return p_name[0].text

        p_name = getProductName()  # assign name to p_name

        def getImgLink():
            img_link = soup.findAll("img",attrs={"alt":p_name})
            return img_link[0]['src']

        def getAmazonRate():
            p_rate = soup.findAll("i",attrs={"data-hook":"average-star-rating"})
            rate = p_rate[0].text.split(" ")[0][0]
            return rate

        r_count = str(getReviewCount())
        p_img_link = getImgLink()
        p_rate = getAmazonRate()

        print("Product Name: " + p_name)
        print("Image Link: " + p_img_link)
        print("Review Count: "+ r_count)
        print("Average Rating: "+ p_rate)
        product = Product(name = p_name, url = url_entered, image = p_img_link, reviewcount = r_count,star = p_rate)
        product.save()

        # ---------------------------------------------------------
        # Handle Reviews Here
        # ---------------------------------------------------------
        def getReviews(url):

            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',}
            req = requests.post(url, headers=headers)
            req.raise_for_status()

            soup = BeautifulSoup(req.content,'lxml')

            r_star = soup.findAll("i",attrs={"data-hook":"review-star-rating"})
            r_title = soup.findAll("a",attrs={"data-hook":"review-title"})
            r_body = soup.findAll("span",attrs={"data-hook":"review-body"})

            for s,t,b in zip(r_star,r_title,r_body):
                global fake_count,review_star_sum, review_count
                global text_str
                review_count+=1
                text_str+=b.text

                blob = TextBlob(b.text)
                foundStar = functions.findStar(blob.polarity)

                star = s.text[0].split(".")
                print(str(review_count) + "------------------------------------------------------------------")
                print("Star : " + str(int(star[0])))
                print("Review title: " + t.text)
                print("Review body: " + b.text)

                if(star!=0 and foundStar!=0):
                    if(functions.isFake(int(star[0]),foundStar)):
                        fake_count+=1
                        print("Is Fake")
                    else:
                        review_star_sum += int(star[0])
                        print("Is Not Fake")


                star, foundStar = 0,0
            print("Number of fake reviews detected is " + str(fake_count))
        # -------------------------------------------------------------
        # Change page to scrape all reviews
        # -------------------------------------------------------------

        p_count = math.ceil(float(getReviewCount())/10) # REVIEW page count
        list_sec = [0.1,0.2,0.3,0.4,0.5,0.6]                        # wait rand seconds between pages

        global fake_count,review_star_sum, review_count
        global text_str

        for p in range(1,p_count+1):
            percent = str(math.ceil(p*100/p_count))
            print("Percent: "+percent)
            getReviews(url+str(p))
            time.sleep(random.choice(list_sec))

        text_sum = summarize.run_summarization(text_str)
        print(" Finished!!! ")
        elapsed_time = time.time()-start_time
        print("Execution Time: "+str(elapsed_time))
        #name shorten
        if len(p_name) > 40:
            p_name_desc = p_name[:40] + '...'
        else:
            p_name_desc = p_name

        context = {
            'p_summary':text_sum,
            'p_name':p_name,
            'p_name_desc':p_name_desc,
            'p_rate':p_rate,
            'p_found_rate':round(review_star_sum/(review_count-fake_count),1),
            'p_review_count':r_count,
            'p_fakereview_count':fake_count,
            'p_url':url_entered,
            'p_img_link':p_img_link,
            }
        text_str = ""
        fake_count,review_star_sum, review_count = [0,0,0]
        return render(request, 'analyzer/home.html',context)
    print("I am here")
    return render(request, 'analyzer/home.html')
