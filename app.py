import mysql.connector
from flask import Flask, render_template, redirect, url_for, request, flash, session
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import config

# Models & Services
from models.data_service import (
    COUNTRIES_BY_CONTINENT,
    FOOD_CATEGORIES,
    predict_and_plot
)
from models.mitigation import MITIGATION_TIPS

# MySQL database configuration (update with your credentials)
db_config = {
    'host':     'localhost',
    'user':     'root',
    'password': 'mysql@123',
    'database': 'plasticheck'
}

app = Flask(__name__)
app.config.from_object(config)

# Flask-Login setup
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id_, name, email, password_hash):
        self.id = id_
        self.name = name
        self.email = email
        self.password_hash = password_hash

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM user WHERE id = %s", (user_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if not row:
        return None
    return User(row['id'], row['name'], row['email'], row['password_hash'])

@app.context_processor
def inject_globals():
    return {'current_year': datetime.now().year}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/effects')
def effects():
    return render_template('effects.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    next_page = request.args.get('next')
    if request.method == 'POST':
        name     = request.form['name']
        email    = request.form['email']
        password = request.form['password']
        pw_hash  = generate_password_hash(password)

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id FROM user WHERE email = %s", (email,))
        if cursor.fetchone():
            error = 'exists'
        else:
            cursor.execute(
                "INSERT INTO user (name, email, password_hash) VALUES (%s, %s, %s)",
                (name, email, pw_hash)
            )
            conn.commit()
            user_id = cursor.lastrowid
            cursor.close()
            conn.close()
            user = User(user_id, name, email, pw_hash)
            login_user(user)
            session.permanent = False  # Session expires on browser close
            return redirect(next_page or url_for('index'))

        cursor.close()
        conn.close()

    return render_template('signup.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    next_page = request.args.get('next')
    if request.method == 'POST':
        email    = request.form['email']
        password = request.form['password']

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM user WHERE email = %s", (email,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row and check_password_hash(row['password_hash'], password):
            user = User(row['id'], row['name'], row['email'], row['password_hash'])
            login_user(user)
            session.permanent = False  # Session expires on browser close
            # Redirect to next page if present, else to index
            return redirect(next_page or url_for('index'))

        error = 'Invalid email or password. Please try again.'

    return render_template('login.html', error=error)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()  # Clear session on logout
    return redirect(url_for('index'))

@app.route('/checker', methods=['GET', 'POST'])
@login_required
def checker():
    if request.method == 'POST':
        continent  = request.form['continent']
        country    = request.form['country']
        categories = request.form.getlist('categories')

        result = predict_and_plot(continent, country, categories)
        result['mitigation'] = MITIGATION_TIPS.get(result['regime'], [])

        # Save to DB (add all paths and info you want)
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO consumption_record
               (user_id, country, selected_categories, timestamp, year, regime, percentages,
                box1_path, box2_path, imp_path, pie_path)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                current_user.id,
                country,
                ",".join(categories),
                datetime.now(),
                2025,
                result['regime'],
                str(result['percentages']),
                result['box1_path'],
                result['box2_path'],
                result['imp_path'],
                result['pie_path']
            )
        )
        conn.commit()
        cursor.close()
        conn.close()

        return render_template(
            'checker.html',
            continents=COUNTRIES_BY_CONTINENT.keys(),
            countries=COUNTRIES_BY_CONTINENT,
            food_categories=FOOD_CATEGORIES,
            result=result
        )

    return render_template(
        'checker.html',
        continents=COUNTRIES_BY_CONTINENT.keys(),
        countries=COUNTRIES_BY_CONTINENT,
        food_categories=FOOD_CATEGORIES
    )

if __name__ == '__main__':
    app.run(debug=True)