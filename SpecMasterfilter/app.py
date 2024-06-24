from flask import Flask, render_template, request

app = Flask(__name__)

# Define some sample laptop data
import sqlite3

def fetch_data_from_db():
    # Connect to the SQLite database
    connection = sqlite3.connect('demo1.db')
    cursor = connection.cursor()

    # Execute a query to fetch all data from the table
    cursor.execute("SELECT * FROM 'dataset'")

    

    # Get column names from cursor description
    columns = [description[0] for description in cursor.description]

    # Fetch all rows from the table
    rows = cursor.fetchall()

    # Convert each row to a dictionary
    data_list = []
    for row in rows:
        row_dict = {columns[i]: row[i] for i in range(len(columns))}
        data_list.append(row_dict)

    # Close the connection to the database
    connection.close()

    return data_list

# Usage example
laptops=fetch_data_from_db()

@app.route('/')
def home():
    return render_template('filter.html')

@app.route('/results', methods=['POST'])
def results():
    selected_brand = request.form.getlist('Brand')
    selected_price = request.form.getlist('Price')
    selected_processor = request.form.getlist('Processor')
    selected_ram = request.form.getlist('Ram')
    selected_storagetype = request.form.getlist('Storage Type')
    selected_storage = request.form.getlist('Rom')
    selected_graphicscardbrand = request.form.getlist('Graphics Card Brand')
    selected_graphicscardtype = request.form.getlist('Graphics Card Type')
    
    selected_price2=[]
    filtered_laptops = []
    if len(selected_price)==1:
        selected_price1 = selected_price[0].split("-")
        for laptop in laptops:
         if(laptop['brand'] in selected_brand and (laptop['Price'] >=int(selected_price1[0]) and laptop['Price'] <=int(selected_price1[1])) and laptop['processor_tier'] in selected_processor and str(laptop['ram_memory']) in selected_ram and laptop['primary_storage_type'] in selected_storagetype and str(laptop['primary_storage_capacity']) in selected_storage and laptop['gpu_brand'] in selected_graphicscardbrand and laptop['gpu_type'] in selected_graphicscardtype) :
            filtered_laptops.append(laptop)
    if len(selected_price)>1:
        for i in range (0,len(selected_price)):
         selected_price1 = selected_price[i].split("-")
         selected_price2.append(selected_price1[0])
         selected_price2.append(selected_price1[1])

         
        for laptop in laptops:
         for i in range(0,len(selected_price2)-1):
            if(laptop['brand'] in selected_brand and (laptop['Price'] >=int(selected_price2[i]) and laptop['Price'] <=int(selected_price2[i+1])) and laptop['processor_tier'] in selected_processor and str(laptop['ram_memory']) in selected_ram and laptop['primary_storage_type'] in selected_storagetype and str(laptop['primary_storage_capacity']) in selected_storage and laptop['gpu_brand'] in selected_graphicscardbrand and laptop['gpu_type'] in selected_graphicscardtype) :
                filtered_laptops.append(laptop)

    return render_template('results.html', laptops=filtered_laptops)

if __name__ == '__main__':
    app.run(debug=True)
