# Flask CRM Application
import atexit
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from functools import wraps
from flask_session import Session

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///crm.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
'''
app = Flask(__name__)
app.secret_key = os.urandom(24) 
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_for_testing')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///crm.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False'''

#yes
# db = SQLAlchemy(app)

app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session"
app.config["SESSION_PERMANENT"] = False  # Sessions clear when browser closes

# Initialize Flask-Session
Session(app)

# Models

class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    #role=  db.Column(db.String(50), nullable=False, default='user')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Customer(db.Model):

    interactions = db.relationship('Interaction', backref='customer', lazy=True, cascade="all, delete-orphan")

    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    notes = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='lead')  # ✅ Ensure this is a STRING
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.String(20), db.ForeignKey('user.id'), default='admin',nullable=False)

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    interaction_type = db.Column(db.String(20), nullable=False)  # call, email, meeting
    summary = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
 
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    due_date = db.Column(db.DateTime, nullable=True)
    completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    assigned_to = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_by= db.Column(db.String(20), default='admin', nullable=True)
    status= db.Column(db.String(20), default= 'Uncompleted')

    def __repr__(self):
        return f'<Task {self.title}>'

# models.py (using SQLAlchemy)
class Product(db.Model):
    __tablename__ = 'product'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Float, nullable=False)
    sku = db.Column(db.String(50), unique=False)
    category = db.Column(db.String(50))
    stock = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    #sale_items = db.relationship('SaleItem', backref='product', lazy=True)

class Sale(db.Model):
    __tablename__ = 'sale'
    sale_id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # ✅ Changed `id` to `sale_id`
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    sale_date = db.Column(db.DateTime, default=datetime.utcnow)
    total_amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default='New') 
    product_ID= db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    # Relationship with Customer
    customer = db.relationship('Customer', backref=db.backref('sales', lazy=True))

'''class SaleItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sale_id = db.Column(db.Integer, db.ForeignKey('sale.sale_id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=1)
    price_at_sale = db.Column(db.Float, nullable=False)
    
    # Relationships
    sale = db.relationship('Sale', backref=db.backref('items', lazy=True))
    product = db.relationship('Product')'''

class SaleItem(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    sale_id = db.Column(db.Integer, db.ForeignKey('sale.sale_id'), nullable=False)  # Refers to `sale_id`
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=1)
    price_at_sale = db.Column(db.Float, nullable=False)
    sale = db.relationship('Sale', backref=db.backref('items', lazy=True))
    product = db.relationship('Product')

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('You need to be logged in to view this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Initialize the app
with app.app_context():
    db.create_all()
    
    # Create admin user if not exists
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(username='admin', email='admin@example.com')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        
# Authentication Routes
#@app.route('/login', methods=['GET', 'POST'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        print(f"Email: {email}, Password: {password}")  # Debugging
        
        user = User.query.filter_by(email=email).first()
        print(f"User found: {user}")  # Debugging
        
        if user:
            print(f"Stored password hash: {user.password_hash}")  # Debugging
            print(f"Password match: {user.check_password(password)}")  # Debugging
        
        if user and user.check_password(password):  # Use the check_password method
            session['user_id'] = user.id
            session['user_name'] = user.username  # Use 'username' instead of 'name'
            #session['user_role'] = user.role  # Ensure 'role' exists in your model
            print(f"Session after login: {session}")  # Debugging
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        
        flash('Invalid email or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')  # Assuming the form field is still named 'name'
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        # Create new user with the correct parameters
        new_user = User(username=username, email=email)
        new_user.set_password(password)  # Use the method to set password
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# Dashboard Route
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login to access the dashboard', 'danger')
        return redirect(url_for('login'))

    # Fetch counts based on status
    customers_count = Customer.query.count()
    leads_count = Customer.query.filter_by(status='lead').count()
    prospects_count = Customer.query.filter_by(status='prospect').count()
    customers_active_count = Customer.query.filter_by(status='customer').count()

    return render_template('dashboard.html',
                           customers_count=customers_count,
                           leads_count=leads_count,
                           prospects_count=prospects_count,
                           customers_active_count=customers_active_count)

# Customer Routes
@app.route('/customers')
def customers():
    if 'user_id' not in session:
        flash('Please login to view customers', 'danger')
        return redirect(url_for('login'))
    
    customers = Customer.query.all()
    return render_template('customers.html', customers=customers)

@app.route('/customer/delete/<int:customer_id>', methods=['POST'])
def delete_customer(customer_id):
    customer = Customer.query.get_or_404(customer_id)
    db.session.delete(customer)
    db.session.commit()
    flash('Customer deleted successfully!', 'success')
    return redirect(url_for('customers'))  # Change to your actual customer list route

@app.route('/customers/add', methods=['GET', 'POST'])
def add_customer():
    if 'user_id' not in session:
        flash('Please login to add customers', 'danger')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
    
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        notes = request.form.get('notes', '').strip()
        status = request.form.get('status', '').strip()

        status = request.form.get('status', 'lead').strip()
        print(f"DEBUG: Type of status: {type(status)}, Value: {status}")

        
        new_customer = Customer(
            first_name = first_name,
            last_name= last_name,
            email= email,
            phone= phone,
            notes= notes,
            status= status,
            created_by= session['user_id']
        )
        
        db.session.add(new_customer)
        db.session.commit()
        
        flash('Customer added successfully', 'success')
        return redirect(url_for('customers'))
    
    return render_template('add_customer.html')

# Task Manager Routes
@app.route('/tasks')
def tasks():
    if 'user_id' not in session:
        flash('Please login to access tasks', 'danger')
        return redirect(url_for('login'))
    return render_template('tasks/tasks.html')

@app.route('/tasks/manage')
def task_manager():
    if 'user_id' not in session:
        flash('Please login to access tasks', 'danger')
        return redirect(url_for('login'))
    tasks = Task.query.all()
    return render_template('tasks/tasks.html', tasks=tasks)

@app.route('/task/add', methods=['POST'])
def add_task():
    title = request.form.get('title')
    description = request.form.get('description')
    due_date = request.form.get('due_date')
    if due_date:
        due_date = datetime.strptime(due_date, '%Y-%m-%d')

    new_task = Task(title=title, description=description, due_date=due_date)
    db.session.add(new_task)
    db.session.commit()
    flash('Task added successfully!', 'success')
    return redirect(url_for('task_manager'))

@app.route('/task/complete/<int:task_id>')
def complete_task(task_id):
    task = Task.query.get(task_id)
    if task:
        task.completed = not task.completed
        db.session.commit()
        flash('Task status updated!', 'info')
    return redirect(url_for('task_manager'))

@app.route('/task/delete/<int:task_id>')
def delete_task(task_id):
    task = Task.query.get(task_id)
    if task:
        db.session.delete(task)
        db.session.commit()
        flash('Task deleted successfully!', 'danger')
    return redirect(url_for('task_manager'))

@app.route('/')
def index():
    return redirect(url_for('login'))

# sales routes
@app.route('/sales')
@login_required
def sales_list():
    sales = Sale.query.order_by(Sale.sale_date.desc()).all()
    return render_template('sales/list.html', sales=sales)

@app.route('/sales/<int:sale_id>')
@login_required
def sale_details(sale_id):
    sale = Sale.query.get_or_404(sale_id)
    return render_template('sales/details.html', sale=sale)

@app.route('/customers/<int:customer_id>/sales')
@login_required
def customer_sales(customer_id):
    customer = Customer.query.get_or_404(customer_id)
    return render_template('sales/customer_sales.html', customer=customer)

@app.route('/sales/new', methods=['GET', 'POST'])
@login_required
def new_sale():
    if request.method == 'POST':
        customer_id = request.form.get('customer_id')
        product_ids = request.form.getlist('product_id[]')  
        quantities = request.form.getlist('quantity[]')

        if not customer_id or not product_ids:
            flash('Customer and at least one product are required.', 'danger')
            return redirect(url_for('new_sale'))

        # Create a new sale record
        sale = Sale(customer_id=customer_id, total_amount=0.0)
        db.session.add(sale)
        db.session.flush()  # Get `sale_id` before committing

        total_amount = 0
        for i in range(len(product_ids)):
            product = Product.query.get(product_ids[i])
            quantity = int(quantities[i])

            if product:
                sale_item = SaleItem(
                    sale_id=sale.sale_id,  # Use `sale.sale_id` instead of `sale.id`
                    product_id=product.id,
                    quantity=quantity,
                    price_at_sale=product.price
                )
                db.session.add(sale_item)
                total_amount += product.price * quantity

        sale.total_amount = total_amount  
        db.session.commit()

        flash('Sale recorded successfully!', 'success')
        return redirect(url_for('sales_list'))
    
    customers = Customer.query.all()
    products = Product.query.all()
    return render_template('sales/new.html', customers=customers, products=products)

from sklearn.cluster import KMeans
# Flask Route for Customer Segmentation
'''@app.route('/customer-segmentation')
@login_required
def customer_segmentation():
    # Fetch sales data
    data = db.session.query(Sale.customer_id, db.func.sum(Sale.total_amount).label("total_spent"))\
                     .group_by(Sale.customer_id).all()
    
    if not data:
        return "No sales data available for segmentation."
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["customer_id", "total_spent"])
    
    # Apply K-Means Clustering (3 segments)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["segment"] = kmeans.fit_predict(df[["total_spent"]])

    # Map segments to human-readable labels
    df["segment_label"] = df["segment"].map({0: "Low Spender", 1: "Medium Spender", 2: "High Spender"})

    # Fetch customer names and emails
    customers = Customer.query.filter(Customer.id.in_(df["customer_id"])).all()
    customer_dict = {c.id: (c.first_name, c.email) for c in customers}
    
    # Add customer info to results
    df["customer_name"] = df["customer_id"].map(lambda cid: customer_dict.get(cid, ("Unknown", ""))[0])
    df["customer_email"] = df["customer_id"].map(lambda cid: customer_dict.get(cid, ("", ""))[1])

    # Convert to dictionary for template
    segmented_data = df.to_dict(orient="records")

    return render_template("customer_segmentation.html", segmented_data=segmented_data)'''
@app.route('/customer-segmentation')
@login_required
def customer_segmentation():
    # Fetch sales data
    data = db.session.query(Sale.customer_id, db.func.sum(Sale.total_amount).label("total_spent"))\
                     .group_by(Sale.customer_id).all()
    
    if not data:  
        return render_template("customer_segmentation.html", segmented_data=None)  # Ensures no raw text output

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["customer_id", "total_spent"])

    # Prevent error if there's only one unique value
    if len(df) < 3:
        return render_template("customer_segmentation.html", segmented_data=None)

    # Apply K-Means Clustering (3 segments)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["segment"] = kmeans.fit_predict(df[["total_spent"]])

    # Map segments to human-readable labels
    df["segment_label"] = df["segment"].map({0: "Low Spender", 1: "Medium Spender", 2: "High Spender"})

    # Fetch customer names and emails
    customers = Customer.query.filter(Customer.id.in_(df["customer_id"])).all()
    customer_dict = {c.id: (c.first_name, c.email) for c in customers}

    # Add customer info to results
    df["customer_name"] = df["customer_id"].map(lambda cid: customer_dict.get(cid, ("Unknown", ""))[0])
    df["customer_email"] = df["customer_id"].map(lambda cid: customer_dict.get(cid, ("", ""))[1])

    # Convert to dictionary for template
    segmented_data = df.to_dict(orient="records")

    return render_template("customer_segmentation.html", segmented_data=segmented_data)



@app.route('/products')
@login_required
def products_list():
    products = Product.query.order_by(Product.name).all()
    return render_template('products/list.html', products=products)

@app.route('/products/<int:product_id>')
@login_required
def product_details(product_id):
    product = Product.query.get_or_404(product_id)
    return render_template('products/details.html', product=product)

@app.route('/products/new', methods=['GET', 'POST'])
@login_required
def new_product():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        price = float(request.form.get('price'))
        sku = request.form.get('sku')
        category = request.form.get('category')
        stock = int(request.form.get('stock', 0))
        
        # Check if SKU already exists
        if sku and Product.query.filter_by(sku=sku).first():
            flash('A product with this SKU already exists', 'danger')
            return redirect(url_for('new_product'))
            
        product = Product(
            name=name,
            description=description,
            price=price,
            sku=sku,
            category=category,
            stock=stock
        )
        
        db.session.add(product)
        db.session.commit()
        
        flash('Product added successfully!', 'success')
        return redirect(url_for('products_list'))
        
    return render_template('products/new.html')

@app.route('/products/<int:product_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_product(product_id):
    product = Product.query.get_or_404(product_id)
    
    if request.method == 'POST':
        product.name = request.form.get('name')
        product.description = request.form.get('description')
        product.price = float(request.form.get('price'))
        product.category = request.form.get('category')
        product.stock = int(request.form.get('stock', 0))
        
        # Only update SKU if it's changed and doesn't conflict
        new_sku = request.form.get('sku')
        if new_sku != product.sku:
            if Product.query.filter_by(sku=new_sku).first():
                flash('A product with this SKU already exists', 'danger')
                return redirect(url_for('edit_product', product_id=product.id))
            product.sku = new_sku
        
        db.session.commit()
        
        flash('Product updated successfully!', 'success')
        return redirect(url_for('product_details', product_id=product.id))
        
    return render_template('products/edit.html', product=product)

@app.route('/products/<int:product_id>/delete', methods=['POST'])
@login_required
def delete_product(product_id):
    product = Product.query.get_or_404(product_id)
    
    # Check if product is used in any sales
    # if SaleItem.query.filter_by(product_id=product.id).first():
    #     flash('Cannot delete product because it is used in sales', 'danger')
    #     return redirect(url_for('product_details', product_id=product.id))
    
    db.session.delete(product)
    db.session.commit()
    
    flash('Product deleted successfully!', 'success')
    return redirect(url_for('products_list'))

def create_tables():
    with app.app_context():
        # Drop all tables and recreate them
        db.drop_all()  # Be careful with this in production!
        db.create_all()

import shutil

def clear_session_files():
    session_dir = app.config["SESSION_FILE_DIR"]
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)  # Delete all session files
        print("All sessions cleared on shutdown.")

# Register the function to execute on shutdown
atexit.register(clear_session_files)
        
'''def db_path():
    return f"Database Path: {app.config['SQLALCHEMY_DATABASE_URI']}"
print(db_path())'''

#prediction model
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import random

class Offer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=True)  # Nullable for general offers
    category = db.Column(db.String(100), nullable=False)
    discount = db.Column(db.Float, nullable=False)
    start_date = db.Column(db.DateTime, default=datetime.utcnow)
    end_date = db.Column(db.DateTime, nullable=True)  # Nullable for indefinite offers
    is_active = db.Column(db.Boolean, default=True)

    customer = db.relationship('Customer', backref=db.backref('offers', lazy=True))

    def __repr__(self):
        return f"<Offer {self.category} - {self.discount}%>"

@app.route('/predict_sales')
def predict_sales():
    sales_data = db.session.query(
        SaleItem.product_id, 
        db.func.sum(SaleItem.quantity).label('total_quantity')
    ).group_by(SaleItem.product_id).all()

    if not sales_data:
        return render_template('predict_sales.html', sales_data=[], prediction=None, best_selling_product_id=None, graph_url=None)

    df = pd.DataFrame(sales_data, columns=['product_id', 'total_quantity'])
    df['future_sales'] = df['total_quantity'].apply(lambda x: x * np.random.uniform(1.1, 1.5))

    # Get the best-selling product
    best_selling_product = df.loc[df['future_sales'].idxmax()]
    best_selling_product_id = int(best_selling_product['product_id'])
    prediction = round(best_selling_product['future_sales'], 2)

    # Generate a bar chart
    fig, ax = plt.subplots()
    ax.bar(df['product_id'].astype(str), df['future_sales'], color='skyblue')
    ax.set_title('Predicted Future Sales')
    ax.set_xlabel('Product ID')
    ax.set_ylabel('Predicted Sales')

    # Convert plot to base64 string for embedding in HTML
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    graph_url = base64.b64encode(img_io.getvalue()).decode()

    return render_template('predict_sales.html', sales_data=df.to_dict(orient='records'), 
                           prediction=prediction, best_selling_product_id=best_selling_product_id, 
                           graph_url=graph_url)

from flask_mail import Mail, Message
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib


app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_password'

mail= Mail(app)

'''@app.route('/offers')
def offers():
    user_id = session.get('user_id')
    if not user_id:
        flash("Please log in to view offers.", "warning")
        return redirect(url_for('login'))
    
    customer = Customer.query.get(user_id)
    if not customer:
        flash("Customer not found.", "danger")
        return redirect(url_for('dashboard'))
    
    # Get categories the customer has purchased from
    customer_sales = Sale.query.filter_by(customer_id=customer.id).all()
    purchased_categories = list(set(sale.category for sale in customer_sales))

    # Get personalized offers
    personalized_offers = Offer.query.filter(Offer.customer_id == customer.id, Offer.is_active == True).all()

    # Get general offers based on the customer's purchase categories
    category_offers = Offer.query.filter(Offer.category.in_(purchased_categories), Offer.customer_id == None, Offer.is_active == True).all()

    return render_template('offers.html', customer=customer, personalized_offers=personalized_offers, category_offers=category_offers)'''

@app.route('/offers')
def show_offers():
    # Fetch all ongoing offers
    offers = Offer.query.all()

    # Fetch offers based on customer purchases
    customer_offers = db.session.query(
        Customer.id.label("customer_id"),
        Customer.first_name.label("customer_name"),
        Customer.email.label("customer_email"),
        Product.category.label("category"),
        Offer.discount.label("discount")
    ).join(Sale, Sale.customer_id == Customer.id) \
     .join(SaleItem, SaleItem.sale_id == Sale.sale_id) \
     .join(Product, SaleItem.product_id == Product.id) \
     .join(Offer, Offer.category == Product.category) \
     .distinct().all()

    return render_template("offers.html", offers=offers, customer_offers=customer_offers)


def generate_discount(customer_id):
    """Generate a discount between 15-20% based on the customer's purchase history."""
    
    purchased_categories = db.session.query(Product.category).join(SaleItem, Product.id == SaleItem.product_id)\
        .join(Sale, SaleItem.sale_id == Sale.id).filter(Sale.customer_id == customer_id).distinct().all()

    if not purchased_categories:
        return None, None  # No previous purchases

    categories = [cat[0] for cat in purchased_categories]

    # Find available inventory in those categories
    available_products = db.session.query(Product.id, Product.name, Product.price).filter(Product.category.in_(categories)).all()

    if not available_products:
        return None, None  # No matching inventory

    # Select a random product and apply a discount (15-20%)
    product = random.choice(available_products)
    discount = round(random.uniform(15, 20), 2)
    discounted_price = round(product.price * (1 - discount / 100), 2)

    return {
        'product_id': product.id,
        'product_name': product.name,
        'original_price': product.price,
        'discount': f"{discount}%",
        'discounted_price': discounted_price
    }, product.category


def send_offer_email(customer, category, discount):
    sender_email = "your-email@example.com"  # Replace with your email
    sender_password = "your-email-password"  # Replace with your email password
    recipient_email = customer.email

    subject = "Exclusive Offer Just for You!"
    body = f"""
    Hello {customer.name},

    We have an exclusive offer for you! Since you love buying from the {category} category, we are offering you a special {discount}% discount on your next purchase.

    Hurry! Grab the deal before it's gone.

    Best regards,
    Your Company
    """

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)  # Use your email provider's SMTP server
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"Offer email sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

@app.route('/send_auto_offers')
def send_auto_offers():
    customers = Customer.query.all()

    for customer in customers:
        purchases = Sale.query.filter_by(customer_id=customer.id).all()
        if purchases:
            category = max(set([p.category for p in purchases]), key=[p.category for p in purchases].count)
            discount = generate_discount()

            new_offer = Offer(customer_id=customer.id, category=category, discount=discount)
            db.session.add(new_offer)
            db.session.commit()

            send_offer_email(customer, category, discount)

    flash("Automated Offers Sent!", "success")
    return redirect(url_for('offers'))


@app.route('/send_manual_offer', methods=['POST'])
def send_manual_offer():
    customer_id = request.form.get('customer_id')
    category = request.form.get('category')
    discount = generate_discount()

    customer = Customer.query.get(customer_id)
    if customer:
        new_offer = Offer(customer_id=customer.id, category=category, discount=discount)
        db.session.add(new_offer)
        db.session.commit()

        send_offer_email(customer, category, discount)

        flash(f"Manual Offer Sent to {customer.name}!", "success")
    else:
        flash("Customer Not Found!", "danger")

    return redirect(url_for('offers'))

'''
@app.route('/predict_sales', methods=['GET'])
def predict_sales():
    sales_data = db.session.query(
        SaleItem.product_id, 
        db.func.sum(SaleItem.quantity).label('total_quantity')
    ).group_by(SaleItem.product_id).all()

    if not sales_data:
        return jsonify({'error': 'No sales data available'})

    df = pd.DataFrame(sales_data, columns=['product_id', 'total_quantity'])
    df['future_sales'] = df['total_quantity'].apply(lambda x: x * np.random.uniform(1.1, 1.5))

    predictions = df.to_dict(orient='records')
    return jsonify(predictions)'''

# Create tables and default admin user
with app.app_context():
    db.create_all()
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(username='admin', email='admin@example.com')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()

#webview.create_window('crm', 'https://127.0.0.0.1')

if __name__ == '__main__':
    app.run(debug=True)

from flask_migrate import Migrate
migrate= Migrate(app,db)

import sys
if getattr(sys, 'frozen', False):
    # If the application is running as a bundled executable
    import os
    os.chdir(sys._MEIPASS)
