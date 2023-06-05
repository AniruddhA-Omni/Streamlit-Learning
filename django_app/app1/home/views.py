from django.shortcuts import render, HttpResponse

# Create your views here.
def index(request):
    context={
        'variable1': 'This is sent',
        'variable2': 'This is also sent',
        'variable3': 'This is Ani'
    }
    return render(request, 'index.html',context)

def about(request):
    return HttpResponse("This is the about page")

def services(request):
    return HttpResponse("This is the services page")

def contact(request):
    return HttpResponse("This is the contact page")