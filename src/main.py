import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import random
import torch 
import torch.nn as nn
from torchvision import transforms



# initialize window
window = tk.Tk()
window.title('PicDetective')
window.geometry("1000x650")

# window title
header = tk.Label(window, text=  "Welcome to PicDetective", font = ( "Arial", 16, "bold"))
header.pack(pady=10)


# splitting the screen into two halves
left = tk.Frame(window, width= 500, height =450)
left.pack(side="left", fill="both", expand=True)

right = tk.Frame(window, width = 500, height=450)
right.pack(side= "right", fill= "both", expand=True)


# place to show the photo
img_label = tk.Label(left)
img_label.pack(expand=True)

# starting scores
plyr_score = 0
comp_score = 0


# text for scores
plyr_score_text = tk.Label(right, text=f"Player: {plyr_score}", font=('Arial', 14))
plyr_score_text.pack(pady=5)

comp_score_text = tk.Label(right, text=f"Computer: {comp_score}", font=('Arial', 14))
comp_score_text.pack(pady=5)


# message at the bottom
result_text = tk.Label(right, text="Pick Real or AI Generated to see computer results", font =("Arial", 11))
result_text.pack(pady=20)

# path to the images
fake_path = r'data/processed/cifake_data/test/FAKE'
real_path = r'data/processed/cifake_data/test/REAL'
photo_paths = []

# gather image paths
if os.path.exists(fake_path):
    for f in os.listdir(fake_path): photo_paths.append(os.path.join(fake_path, f))
if os.path.exists(real_path):
    for f in os.listdir(real_path): photo_paths.append(os.path.join(real_path,f))

# randomize the list
random.shuffle(photo_paths)



# brain for the computer, same as notebook
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),
    nn.Linear(32 * 8 * 8, 128),
    nn.ReLU(),

    nn.Linear(128, 1)
)
# load the model from the file
model.load_state_dict(torch.load('models/PicDetective_model.pth'))
model.eval()


# scale images for the model to read, same as notebook
transformer = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


current_label = ""
current_path = ""
# function to show next photo
def show_next_photo():
    global current_label, current_path
    if len(photo_paths) > 0:
        current_path = photo_paths.pop()
        
        #finds out whether current photo is real or fake (Claude. Prompt: help me get the actual label of the photo)     
        clean_path = current_path.replace("\\", "/")
        if "/FAKE/" in clean_path or "/FAKE" in clean_path.upper():
            current_label = "AI"
        else:
            current_label = "Real"


        # load and resize image for screen
        img = Image.open(current_path).convert("RGB")
        img_display = img.resize((400, 400)) 
        tk_img = ImageTk.PhotoImage(img_display)
        img_label.config(image=tk_img)
        img_label.image = tk_img



# function to check guesses
def check_guess(user_choice):
    global plyr_score, comp_score
    


    # check if player got it right
    if user_choice == current_label:
        plyr_score += 1
        msg = "You were correct! "
    else:
        msg = "You were wrong! "


    # make the computer guess using the brain
    img = Image.open(current_path).convert("RGB")
    img_tensor = transformer(img).unsqueeze(0) 
    
    with torch.no_grad():
        output = model(img_tensor).squeeze()
        prob = torch.sigmoid(output)
        
        # labeling computer predictions (Claude. Prompt: how to turn model output into a label)
        if prob > 0.5:
            comp_choice = "Real"
        else:
            comp_choice = "AI"


    # check if computer got it right
    if comp_choice == current_label:
        comp_score += 1
        msg += "The Computer was correct."
    else:
        msg += "The Computer was wrong."


    # update text and scores on screen
    result_text.config(text=msg)
    plyr_score_text.config(text=f"Player: {plyr_score}")
    comp_score_text.config(text=f"Computer: {comp_score}")
    
    show_next_photo()



# creating buttons (Claude. Prompt: how to pass arguments to a button command using lambda)
real_btn = tk.Button(right, text = "REAL", width=12, bg= "#4CAF50", fg= "white", font =("Impact", 17), command=lambda: check_guess("Real"))
real_btn.pack(pady=10)
 
ai_btn = tk.Button(right, text = "AI GENERATED", width=12, bg="#f44336", fg = "white", font=("Impact", 17), command=lambda: check_guess("AI"))
ai_btn.pack(pady=10)


# show first photo
show_next_photo()



# function to load your own image from your computer
def load_custom_image():
    # open a file selector
    file_path = filedialog.askopenfilename()
    
    if file_path:
        # show the image on the screen with same code as above
        img = Image.open(file_path).convert("RGB")
        img_display = img.resize((400, 400)) 
        tk_img = ImageTk.PhotoImage(img_display)
        img_label.config(image=tk_img)
        img_label.image = tk_img
        
        #make the computer guess
        img_tensor = transformer(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor).squeeze()
            prob = torch.sigmoid(output).item()
            

            # percentage ai that comp guesses (Claude. Prompt: how to calculate a confidence percentage)
            ai_percent = prob * 100

            # format the message to show the percent
            res = f"The AI is {ai_percent:.1f}% sure this is AI Generated!"
        
        # update the message
        result_text.config(text=res)



# button to upload your own photo
upload_btn = tk.Button(right, text="Load Your Own Photo", width=20, bg="#2196F3", fg="white", font=("Arial", 12, "bold"), command=load_custom_image)
upload_btn.pack(pady=20)

window.mainloop()