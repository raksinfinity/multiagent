#libraries
from tkinter.ttk import *
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt 
import math
import time
class Application(Frame):
	def createWidgets(self):
		self.lbl1=Label(window,text='Algorithm')
		self.lbl1.place(relx = 0.1, rely = .1, anchor ='sw')
		self.lbl2=Label(window,text='No. of rovers')
		self.lbl2.place(relx = 0.1, rely = .15, anchor ='sw')
		self.lbl3=Label(window,text='Hyper parameters')
		self.lbl3.place(relx = 0.1, rely = .2, anchor ='sw')
		self.lbl4=Label(window,text='Nuclear energy')
		self.lbl4.place(relx = 0.1, rely = .25, anchor ='sw')
		self.nbot=5
		#self.bt=[self.nbot]
		self.comb1 = Combobox(window)
		self.comb1['values']= ('Q Learning', 'SARSA')
		self.comb1.current(0) 
		self.comb1.place(relx = 0.2, rely = .1, anchor ='sw')
		algo=self.comb1.get()
		self.widv2 = StringVar()
		self.comb2 = Combobox(window,textvariable=self.widv2)
		self.comb2['values']= (2,3,4,5,6,7,8,9,10)
		self.comb2.current(3)  
		self.comb2.place(relx = 0.2, rely = .15, anchor ='sw')
		self.comb2.bind("<<ComboboxSelected>>", self.cb)

		self.comb3 = Combobox(window)
		self.comb3['values']= ('learning rate=0.5', 'learning rate=0.9')
		self.comb3.current(0)  
		self.comb3.place(relx = 0.2, rely = .2, anchor ='sw')
		lern=self.comb3.get()
		self.comb4 = Combobox(window)
		self.comb4['values']= (220, 196)
		self.comb4.current(0)  
		self.comb4.place(relx = 0.2, rely = .25, anchor ='sw')
		enr=int(self.comb4.get())

		self.btn = Button(window, text="Start", command=self.clicked)
		self.btn.place(relx=.2, rely=.3, anchor='sw')
			
	def clicked(self):
		self.comb1.destroy()
		self.lbl1.destroy()
		self.comb2.destroy()
		self.lbl2.destroy()
		self.comb3.destroy()
		self.lbl3.destroy()
		self.comb4.destroy()
		self.lbl4.destroy()
		self.btn.destroy()
		self.pldron()
		self.can.pack()
	def cb(self,event):
		self.nbot=int(self.comb2.get())
		
	def pldron(self):
		dron=[]
		#cur=self.onrobs[ind]
		stp=2*3.14/self.nbot
		rad=20
		for i in range(int(self.nbot)):
			cx=200+100*np.cos(stp*i)
			cy=200+100*np.sin(stp*i)
			dr=self.can.create_oval(cx-rad,cy-rad,cx+rad,cy+rad,fill='green')
			dron.append(dr)
		for cur in onrobs:  
			[self.can.itemconfig(dron[int(j)], fill='blue') for j in cur]
			window.after(1000)
			self.can.update()
			[self.can.itemconfig(dron[k], fill='green') for k in range(self.nbot)]
			self.can.pack()						


	def __init__(self,canvas):
		#Frame.__init__(self, master)
		self.can=canvas
		#self.can.pack()
		self.createWidgets()

#hyper parameters
def calc():
	ep=.7
	lam=.9
	nbot=5
	on=[12,13,14,15,16,17,18,19,20,21]
	off=[5,5,5,5,5,7,7,7,7,7]
	ofpwr=np.array(off)[:nbot]
	onpwr=np.array(on)[:nbot]
	fulp=np.array([220,196])-np.sum(on[nbot:])-np.sum(off[nbot:])

	#create space state
	voc='0123456789'[:nbot]
	res=[]
	def arr(st):
	  if len(st)==nbot: res.append(st)
	  rest=[s for s in voc if st.find(s)==-1]
	  for i in rest:
	    arr(st+i)
	arr('')    
	spaln=len(res)
	actln=2**nbot
	math.factorial(nbot)-len(res)

	#reward function
	def reward(st0,act,dep,prin):
	  st=res[st0]
	  act=bin(act)[2:]
	  act='0'*(nbot-len(act))+act
	  #ofs=[i for i in range(len(act)) if act[i]=='0']
	  ons=[str(i) for i in range(len(act)) if act[i]=='1']  
	  st1=st[:]
	  inds=[]
	  for i in ons:
	    inds.append(st1.index(i))
	  am=np.argsort(inds)
	  ons=list(np.array(ons)[am])
	  onind=[int(j) for j in ons]  
	  for i in ons:  
	    ind=st1.index(i)
	    st1=st1[:ind]+st1[ind+1:]+st1[ind]
	  pwr=np.sum(onpwr[onind])+np.sum(ofpwr[onind])
	  if pwr<fulp[dep]:
	    rew=int(np.sum([(nbot-st.index(i))*3 for i in ons])*pwr/fulp[dep]) #*pwr/fulp[dep] 
	  else: rew=0
	  st1=''.join(st1)
	  if prin: print('st0 %s, st1 %s, ON %s, reward %d'%(st,st1,ons,rew))
	  st1=res.index(st1)
	  return rew,st1,ons

	#train
	st=0
	dep=0
	sts=[]; acts=[]; onrobs=[]
	qtab=np.ones((spaln,actln),dtype='uint8')
	qplu=[]
	for i in range(actln):
	  bn=bin(i)[2:]
	  oncn=len([j for j in bn if j=='1'])
	  if oncn>nbot//2: qplu.append(i)
	qtab[:,qplu]=2
	for i in range(20000):
	  if not i%1000: print(i,end=' ')
	  mx=np.argmax(qtab[st,:])
	  act=np.random.choice([mx,0],p=[ep,1-ep])
	  if act==0: act=np.random.choice(actln)
	  rew,st1,ons=reward(st,act,dep,prin=0)
	  qtab[st,act]=qtab[st,act]*(1-lam) + lam*rew
	  st=st1
	  sts.append(st)
	  acts.append(act)
	  onrobs.append(len(ons))

	#play
	st=0
	sts=[]
	acts=[]
	onrobs=[]
	for i in range(1000):
	  act=np.argmax(qtab[st,:])
	  rew,st1,ons=reward(st,act,dep,prin=1)
	  st=st1
	  sts.append(st)
	  acts.append(act)
	  onrobs.append(ons)
	return onrobs

window = Tk()
window.title("Life on Mars 2021")
window.geometry('1000x600')
c=Canvas(window, width=1000, height=600)
#c.pack_forget()
onrobs=calc()	
app = Application(c)
window.mainloop()
window.destroy()

