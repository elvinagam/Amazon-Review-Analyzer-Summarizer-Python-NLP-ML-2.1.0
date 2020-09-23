from django import forms
from analyzer.models import Product, Review

class ProductForm(forms.ModelForm):
    url = forms.CharField(max_length=1000,help_text="Enter Product URL")

    class Meta:
        model = Product
        fields = ('url',)
