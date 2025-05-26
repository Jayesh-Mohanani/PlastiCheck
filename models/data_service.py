import os
from datetime import datetime
from models.country_mapping import COUNTRIES_BY_CONTINENT, FOOD_CATEGORIES
from models.predictor import run_prediction

def predict_and_plot(continent, country, categories):
    """
    Calls run_prediction from predictor.py and saves the four figures as PNGs.
    Returns the paths for Flask to render.
    """
    # Call your ML function (city, categories)
    result = run_prediction(country, categories)

    # Unique suffix for this user's run (prevents image overwrite)
    suffix = datetime.now().strftime('%Y%m%d%H%M%S%f')
    outdir = os.path.join(os.path.dirname(__file__), '..', 'static', 'img', 'tmp')
    os.makedirs(outdir, exist_ok=True)

    forecast_path = f'img/tmp/forecast_plot_{suffix}.png'
    selected_path = f'img/tmp/selected_plot_{suffix}.png'
    imp_path      = f'img/tmp/imp_plot_{suffix}.png'
    pie_path      = f'img/tmp/pie_{suffix}.png'

    # Save the matplotlib/plotly figures returned by run_prediction
    result['fig1'].write_image(os.path.join(outdir, f'forecast_plot_{suffix}.png'))
    result['fig2'].write_image(os.path.join(outdir, f'selected_plot_{suffix}.png'))
    result['fig3'].write_image(os.path.join(outdir, f'imp_plot_{suffix}.png'))
    result['fig4'].write_image(os.path.join(outdir, f'pie_{suffix}.png'))


    return {
        'box1_path': forecast_path,
        'box2_path': selected_path,
        'imp_path':  imp_path,
        'pie_path':  pie_path,
        'regime':    result['regime'],
        'percentages': result['percentages'],
        'regime_html': result.get('regime_html', ''),  # HTML-formatted output
        'current_value_2025': round(result.get('current_value_2025', ''), 2),
        'user_country': result.get('user_country', country),
        'alert_msg': result.get('alert_msg', ''),
    }