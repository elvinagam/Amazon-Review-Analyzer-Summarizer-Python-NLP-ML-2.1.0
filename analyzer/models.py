from django.db import models
from django.contrib.auth.models import User
from datetime import datetime


class Product(models.Model):
    name = models.CharField(max_length=1000)
    url = models.CharField(max_length=1000,default="null")
    image = models.CharField(max_length=1000)
    reviewcount = models.IntegerField()
    star = models.CharField(max_length=4,default="null")
    analyze_date = models.DateField(datetime.now(),default='1990-10-10')

    def __str__(self):
        return self.name

class Review(models.Model):
    p_id = models.ForeignKey(Product, on_delete=models.CASCADE)
    r_star = models.CharField(max_length=4)
    r_title = models.TextField()
    r_body = models.TextField()
    isFake = models.BooleanField(default=0)
    def __str__(self):
        return self.r_title
