'''
docstring stuff
'''
import sqlite3
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

def create_tables():
    """Creates the inventory database & tables if not already existing"""

    conn = sqlite3.connect('inventory.db')
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY NOT NULL,
            description TEXT,
            quantity INTEGER DEFAULT 0,
            type TEXT NOT NULL
            
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS bom (
            parent_id TEXT NOT NULL,
            child_id TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            FOREIGN KEY (parent_id) REFERENCES items (id),
            FOREIGN KEY (child_id) REFERENCES items (id)
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    '''Render the index.html route'''
    return render_template('index.html')

@app.route('/create_item', methods=['GET', 'POST'])
def create_item():
    '''Render the create_item.html route'''
    if request.method == 'POST':
        item_id = request.form['item_id']
        item_type = request.form['item_type']
        description = request.form['description']
        quantity = int(request.form['quantity'])

        conn = sqlite3.connect('inventory.db')
        cur = conn.cursor()
        cur.execute('INSERT INTO items (id, description, quantity, type) VALUES (?, ?, ?, ?)', (item_id, description, quantity, item_type))
        conn.commit()
        conn.close()

        return redirect(url_for('index'))
    return render_template('create_item.html')

@app.route('/create_bom', methods=['GET', 'POST'])
def create_bom():
    '''Render the create_bom.html route'''
    error = None
    if request.method == 'POST':
        assembly_id = request.form['assembly_id']
        description = request.form['description']

        conn = sqlite3.connect('inventory.db')
        cur = conn.cursor()

        existing_assembly = cur.execute('SELECT id FROM items WHERE id = ? AND type = "assembly"', (assembly_id,)).fetchone()

        if existing_assembly:
            error = f'Assembly ID {assembly_id} already exists. Please use a different ID.'
        else:
            cur.execute('INSERT INTO items (id, description, type) VALUES (?, ?, "assembly")', (assembly_id, description))
            conn.commit()

            for i in range(1, 6):
                part_id = request.form[f'part_id{i}']
                quantity_str = request.form[f'quantity{i}']

                if part_id and quantity_str:
                    quantity = int(quantity_str)
                    cur.execute('INSERT INTO bom (parent_id, child_id, quantity) VALUES (?, ?, ?)', (assembly_id, part_id, quantity))
                    conn.commit()

            conn.close()

            return redirect(url_for('index'))

    return render_template('create_bom.html', error=error)

@app.route('/load_bom_from_excel', methods=['GET', 'POST'])
def load_bom_from_excel():
    '''Render the load_bom_from_excel.html route'''
    if request.method == 'POST':
        # Implement the functionality to load BOM from Excel
        pass
    return render_template('load_bom_from_excel.html')

@app.route('/view_bom', methods=['GET', 'POST'])
def view_bom():
    '''Renderthe viw_bom.html route'''
    if request.method == 'POST':
        assembly_id = request.form['assembly_id']

        conn = sqlite3.connect('inventory.db')
        cur = conn.cursor()

        top_level_bom = cur.execute('''
            SELECT i.id, i.description, b.quantity, i.quantity
            FROM bom AS b
            JOIN items AS i ON b.child_id = i.id
            WHERE b.parent_id = ?
        ''', (assembly_id,)).fetchall()

        detailed_bom = cur.execute('''
                        SELECT i.id, i.description, SUM(b.quantity), i.quantity
            FROM bom AS b
            JOIN items AS i ON b.child_id = i.id
            GROUP BY i.id
        ''').fetchall()

        conn.close()

        return render_template('bom_result.html', assembly_id=assembly_id, top_level_bom=top_level_bom, detailed_bom=detailed_bom)
    return render_template('view_bom.html')

@app.route('/in_stock')
def in_stock():
    '''Render in_stock.html route'''
    conn = sqlite3.connect('inventory.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM items')
    items = cur.fetchall()
    return render_template('in_stock.html', items=items)

@app.route('/add_assembly_to_inventory', methods=['GET', 'POST'])
def add_assembly_to_inventory():
    '''Rednder add_assemlby_to_inventory.html route'''
    if request.method == 'POST':
        assembly_id = request.form['assembly_id']
        quantity = int(request.form['quantity'])

        conn = sqlite3.connect('inventory.db')
        cur = conn.cursor()

        # Check if assembly exists
        cur.execute('SELECT * FROM items WHERE id=? AND type="assembly"', (assembly_id,))
        assembly = cur.fetchone()
        if not assembly:
            return render_template('add_assembly_to_inventory.html', error='Assembly not found')
        
        # Update items quantity based on assembly's BOM
        cur.execute('SELECT * FROM bom WHERE parent_id=?', (assembly_id,))
        bom_items = cur.fetchall()

        for item in bom_items:
            _, child_id, item_quantity = item
            cur.execute('UPDATE items SET quantity=quantity-?*? WHERE id=?', (item_quantity, quantity, child_id))

        # Update assembly quantity in stock
        cur.execute('UPDATE items SET quantity=quantity+? WHERE id=?', (quantity, assembly_id))
        
        conn.commit()
        return redirect(url_for('in_stock'))

    return render_template('add_assembly_to_inventory.html')


if __name__ == '__main__':
    create_tables()
    app.run(debug=True)
