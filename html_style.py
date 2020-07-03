
fontsize = '18px'


def selected_tab_style():
    return {'borderTop': '1px solid #d6d6d6', 'borderBottom': '1px solid #d6d6d6',
            'backgroundColor': '#119DFF', 'color': 'white', 'padding': '6px',
            'fontWeight': 'bold', 'font-family': 'verdana', 'font-size': '30px'}


def tab_style():
    return {'background-color': 'black', 'font-weight': 'bold',
            'font-family': 'verdana', 'font-size': '30px',
            'color': 'white', 'borderBottom': '1px solid #d6d6d6'}


def stock_analyzer_titles():
    return {'font-family': 'verdana', 'margin-left': 'auto', 'margin-right': 'auto',
            'font-size': '35px', 'display': 'center-block', 'position': 'relative', 'text-align': 'center',
            'padding-left': '0px', 'padding-bottom': '40px', 'width': '1600px',
            'left': '0%',
            'color': 'white', 'text-decoration': 'underline', 'bottom': 0, 'right': 0}


def suggestion_text():
    return {'display': 'center-block', 'font-family': 'verdana', 'padding-left': '120px',
            'padding-bottom': '00px', 'color': 'white', 'display': 'inline-block'}


def tables_styler(width_percentage):
    return {'font-family': 'verdana', 'margin-left': 'auto', 'margin-right': 'auto',
            'font-size': '18px',
            'display': 'center-block', 'position': 'relative', 'align': 'center',
            'width': width_percentage,
            'padding-left': '0px', 'padding-bottom': '25px', 'bottom': 0, 'right': 0}


def multi_table_styler(table_side_width, width_percentage):
    return {'font-family': 'verdana', 'margin-left': 'auto', 'margin-right': 'auto',
            'font-size': fontsize,
            'display': 'inline-block', 'position': 'relative', 'align': 'left',
            'left': table_side_width,
            'padding-left': '10px', 'padding-bottom': '40px', 'bottom': 0, 'right': 0,
            "width": width_percentage}
