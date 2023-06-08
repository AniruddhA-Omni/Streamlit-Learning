from django.shortcuts import render, HttpResponse
from datetime import datetime
from home.models import Contact

# Create your views here.
def index(request):
    context={
        'variable1': 'This is sent',
        'variable2': 'This is also sent',
        'variable3': 'This is Ani'
    }
    return render(request, 'index.html',context)

def about(request):
    return render(request, 'about.html')

def services(request):
    return render(request, 'services.html')

def contact(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone= request.POST.get('phn')
        desc= request.POST.get('desc')
        contact = Contact(name=name, email=email, phone=phone, desc=desc, date=datetime.today())
        contact.save()
        
    return render(request, 'contact.html')