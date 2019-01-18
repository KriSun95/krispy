'''
Functions to go in here (I think!?):
	KC: 19/12/2018, ideas-
	~full_width()			<

    KC: 19/12/2018, added-	
    ~silly_jupyter_notebook()
'''

#make cells the width of the browser
def silly_jupyter_notebook(output=True):
    """
    Use the full width of the page Jupyter!!
    """
    from IPython.core.display import display, HTML
    

    display(HTML('<style>.container{width:100% !important;}</style>'))
    
    if output == True:
        print('Wow, so much space! So much room for activities!')