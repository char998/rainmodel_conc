import ipywidgets as widgets
from IPython.display import display
import datetime
import pytz
import json

# Path to save widget settings
CONFIG_FILE = "widget_settings.json"
local_tz = pytz.timezone('UTC')

# --- Load existing settings ---
def load_settings():
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Defaults if file doesn't exist
        return {
            'theta_type': 'theta_e_reversible',
            'entrainment_beginning': 'from ground',
            'interp_interval': 'No',
            'start_date': '2024-07-01',
            'end_date': '2024-07-16',
            'start_time': '00:00:00',
            'end_time': '23:59:00'
        }

saved_settings = load_settings()

# --- Widgets ---
theta_type_widget = widgets.Dropdown(
    options=['theta_e_reversible', 'theta_l', 'theta_e', 'GB84'],
    value=saved_settings.get('theta_type', 'theta_e_reversible'),
    description='Theta Type:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

entrainment_widget = widgets.Dropdown(
    options=['from ground', 'from LCL'],
    value=saved_settings.get('entrainment_beginning', 'from ground'),
    description='Entrainment:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

interpolation_widget = widgets.Dropdown(
    options=['No','1min', '5min','10min','20min','30min'],
    value=saved_settings.get('interp_interval', 'No'),
    description='Downscaling input data:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)


# Safe parsing helpers
def parse_date_safe(date_str, default):
    try:
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    except Exception:
        return default

def parse_time_safe(time_str, default):
    return time_str if time_str else default

start_date_widget = widgets.DatePicker(
    description='Start Date:',
    value=parse_date_safe(saved_settings.get('start_date'), datetime.date(2024,7,1)),
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='200px')
)

end_date_widget = widgets.DatePicker(
    description='End Date:',
    value=parse_date_safe(saved_settings.get('end_date'), datetime.date(2024,7,16)),
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='200px')
)

start_time_widget = widgets.Text(
    value=parse_time_safe(saved_settings.get('start_time'), '00:00:00'),
    description='Start Time:',
    placeholder='HH:MM:SS',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='200px')
)

end_time_widget = widgets.Text(
    value=parse_time_safe(saved_settings.get('end_time'), '23:59:00'),
    description='End Time:',
    placeholder='HH:MM:SS',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='200px')
)

# --- Apply button ---
apply_button = widgets.Button(
    description='Apply Settings',
    button_style='success',
    tooltip='Click to apply and save parameters',
    layout=widgets.Layout(width='200px')
)

# --- Helpers ---
def format_datetime_string(dt):
    return dt.astimezone(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%S") if dt.tzinfo else dt.strftime("%Y-%m-%dT%H:%M:%S")

def save_settings():
    settings = {
        'theta_type': theta_type_widget.value,
        'entrainment_beginning': entrainment_widget.value,
        'interp_interval': interpolation_widget.value,
        'start_date': start_date_widget.value.isoformat() if start_date_widget.value else None,
        'end_date': end_date_widget.value.isoformat() if end_date_widget.value else None,
        'start_time': start_time_widget.value,
        'end_time': end_time_widget.value
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"❌ Error saving settings: {e}")

# --- Apply handler ---
def on_apply_clicked(b):
    global theta_type, entrainment_beginning, interp_interval, slice_str, slice_tuple

    theta_type = theta_type_widget.value
    entrainment_beginning = entrainment_widget.value
    interp_interval = interpolation_widget.value
    try:
        start_dt = datetime.datetime.combine(
            start_date_widget.value,
            datetime.time(*map(int, start_time_widget.value.split(':')))
        )
        end_dt = datetime.datetime.combine(
            end_date_widget.value,
            datetime.time(*map(int, end_time_widget.value.split(':')))
        )
        start_dt = local_tz.localize(start_dt)
        end_dt = local_tz.localize(end_dt)
    except (ValueError, TypeError) as e:
        print(f"❌ Error parsing date/time: {e}")
        return

    start_str = format_datetime_string(start_dt)
    end_str = format_datetime_string(end_dt)
    slice_str = f"{start_str}, {end_str}"
    slice_tuple = tuple(slice_str.split(", "))

    save_settings()

    print(f"✓ Parameters applied and saved to {CONFIG_FILE}")
    print(f"  theta_type = '{theta_type}'")
    print(f"  entrainment_beginning = '{entrainment_beginning}'")
    print(f"  interp_interval = '{interp_interval}'")
    print(f"  slice_str = '{slice_str}'")
    print(f"  slice_tuple = {slice_tuple}")
    print(f"  Time range: {start_dt.strftime('%Y-%m-%d %H:%M')} → {end_dt.strftime('%Y-%m-%d %H:%M')}")

apply_button.on_click(on_apply_clicked)

# --- Layout ---
date_time_box = widgets.HBox([
    widgets.VBox([start_date_widget, end_date_widget]),
    widgets.VBox([start_time_widget, end_time_widget])
])

widgets_box = widgets.VBox([
    widgets.HTML("<h3>Model Configuration</h3>"),
    widgets.HTML("<p>Select the parameters and time range for your cloud - parcel model:</p>"),
    widgets.HTML("<b>Model Parameters:</b>"),
    theta_type_widget,
    entrainment_widget,
    widgets.HTML("<b>Input data:</b>"),
    interpolation_widget,
    widgets.HTML("<br><b>Time Range Selection:</b>"),
    date_time_box,
    widgets.HTML("<br>"),
    apply_button
])

display(widgets_box)

# --- Load settings for model execution ---
def load_model_settings():
    try:
        with open(CONFIG_FILE, 'r') as f:
            settings = json.load(f)

        theta_type = settings['theta_type']
        entrainment_beginning = settings['entrainment_beginning']
        interp_interval = settings['interp_interval']
        start_dt = datetime.datetime.combine(
            datetime.datetime.strptime(settings['start_date'], '%Y-%m-%d').date(),
            datetime.time(*map(int, settings['start_time'].split(':')))
        )
        end_dt = datetime.datetime.combine(
            datetime.datetime.strptime(settings['end_date'], '%Y-%m-%d').date(),
            datetime.time(*map(int, settings['end_time'].split(':')))
        )
        start_dt = local_tz.localize(start_dt)
        end_dt = local_tz.localize(end_dt)

        start_str = format_datetime_string(start_dt)
        end_str = format_datetime_string(end_dt)
        slice_str = f"{start_str}, {end_str}"
        slice_tuple = tuple(slice_str.split(", "))

        return theta_type, entrainment_beginning, interp_interval, slice_str, slice_tuple, start_dt, end_dt

    except Exception as e:
        print(f"⚠️ Could not read config file: {e}. Using defaults.")
        return 'theta_e_reversible', 'from ground', '2024-07-01T00:00:00, 2024-07-16T23:59:00', ('2024-07-01T00:00:00', '2024-07-16T23:59:00'), None, None

theta_type, entrainment_beginning, interp_interval, slice_str, slice_tuple, start_dt, end_dt = load_model_settings()