from django.shortcuts import render, redirect
from .models import User
from django.http import HttpResponse


# Helper Functions
def is_logged_in(request):
    """Checks if the user is logged in by verifying the session."""
    return 'Id' in request.session


# Authentication Views
def login(request):
    """Handles user login."""
    if is_logged_in(request):
        return redirect('home')

    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Fetch the user from MongoDB by email
        user = User.find_user_by_email(email)

        if user and User.verify_password(user, password):
            # Set session details
            request.session['Role'] = user['Role']
            request.session['Id'] = str(user['_id'])
            return redirect('home')

        # Invalid credentials
        error_message = 'Invalid password' if user else 'User not found'
        return render(request, 'login.html', {'error': error_message})

    # Default GET request
    return render(request, 'login.html')


def register(request):
    """Renders the registration page."""
    if is_logged_in(request):
        return redirect('home')
    return render(request, 'register.html')


def forgot_pass(request):
    """Renders the forgot password page."""
    if is_logged_in(request):
        return redirect('home')
    return render(request, 'forgot_password.html')


# Error Handling Views
def error_404(request, exception):
    """Handles 404 errors."""
    return render(request, 'error_404.html', status=404)


def error_500(request):
    """Handles 500 errors."""
    return render(request, 'error_404.html', status=500)


def error_403(request, exception):
    """Handles 403 errors."""
    return render(request, 'error_404.html', status=403)


def error_400(request, exception):
    """Handles 400 errors."""
    return render(request, 'error_404.html', status=400)
