from tkinter.ttk import *
from tkinter import *
import numpy as np
import math
import matplotlib.pyplot as plt 

class Application(Frame):
	def createWidgets(self):
		self.lbl1=Label(window,text='Algorithm')
		self.lbl1.place(relx = 0.1, rely = .2, anchor ='sw')
		self.lbl2=Label(window,text='No. of rovers')
		self.lbl2.place(relx = 0.1, rely = .25, anchor ='sw')
		self.lbl3=Label(window,text='Hyperparameters')
		self.lbl3.place(relx = 0.1, rely = .3, anchor ='sw')
		self.lbl4=Label(window,text='Nuclear energy')
		self.lbl4.place(relx = 0.1, rely = .35, anchor ='sw')
		self.nbot=5
		self.dep=0
		self.powr = StringVar()
		self.powr.set('196') if self.dep else self.powr.set('220')
		self.lblnu=Label(window,textvariable=self.powr)
		self.lblnu.place_forget()		
		#self.bt=[self.nbot]
		self.comb1 = Combobox(window)
		self.comb1['values']= ('Q Learning', 'SARSA')
		self.comb1.current(0) 
		self.comb1.place(relx = 0.4, rely = .2, anchor ='sw')
		algo=self.comb1.get()
		self.widv2 = StringVar()
		self.comb2 = Combobox(window,textvariable=self.widv2)
		self.comb2['values']= (2,3,4,5,6,7,8,9,10)
		self.comb2.current(3)  
		self.comb2.place(relx = 0.4, rely = .25, anchor ='sw')
		self.comb2.bind("<<ComboboxSelected>>", self.cb)

		self.comb3 = Combobox(window)
		self.comb3['values']= ('learning rate=0.5', 'learning rate=0.9')
		self.comb3.current(0)  
		self.comb3.place(relx = 0.4, rely = .3, anchor ='sw')
		lern=self.comb3.get()
		self.comb4 = Combobox(window)
		self.comb4['values']= (220, 196)
		self.comb4.current(0)  
		self.comb4.place(relx = 0.4, rely = .35, anchor ='sw')
		self.comb4.bind("<<ComboboxSelected>>", self.cb1)
		enr=int(self.comb4.get())

		self.btn = Button(window, text="Start", command=self.clicked)
		self.btn.place(relx=.4, rely=.45, anchor='sw')
			
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
		self.onrobs=calc(self.nbot,self.dep)
		self.lblnu.place(x = 150, y = 180, anchor ='sw')
		self.pldron()
		#self.can.pack()
		
	def cb(self,event):
		self.nbot=int(self.comb2.get())

	def cb1(self,event):
		self.dep=0 if self.comb4.get()=='220' else 1

		
	def pldron(self):
		dron=[]
		#cur=self.onrobs[ind]
		stp=2*3.14/self.nbot
		l=200
		rad=20
		on=[12,13,14,15,16,17,18,19,20,21]
		off=[5,5,5,5,5,7,7,7,7,7]
		ofpwr=np.array(off)[:self.nbot]
		onpwr=np.array(on)[:self.nbot]	
		fulp=int(self.powr.get())-np.sum(on[self.nbot:])-np.sum(off[self.nbot:])
	
		for i in range(int(self.nbot)):
			cx=l+100*np.cos(stp*i)
			cy=l+100*np.sin(stp*i)
			dr=self.can.create_oval(cx-rad,cy-rad,cx+rad,cy+rad,fill='green')
			rec=self.can.create_rectangle(l-rad,l-rad,l+rad,l+rad,fill='yellow')
			dron.append(dr)
		for cur in self.onrobs:  
			lns=[]
			[self.can.itemconfig(dron[int(j)], fill='blue') for j in cur]
			for j in cur:
				ln=self.can.create_line(l+100*np.cos(stp*int(j)),l+100*np.sin(stp*int(j)),200,200) 
				lns.append(ln)
			onind=[int(j) for j in cur]
			pwr=np.sum(onpwr[onind])+np.sum(ofpwr[onind])
			self.powr.set('charge= %d / %d '%(pwr,fulp))	
			self.can.pack()		
			self.can.update()	
			window.after(1000)
			[self.can.itemconfig(dron[k], fill='green') for k in range(self.nbot)]
			[self.can.delete(ll) for ll in lns]
					
	def __init__(self,canvas):
		#Frame.__init__(self, master)
		self.can=canvas
		#self.can.pack()
		self.createWidgets()

#learning 
def calc(nbot,dep):
	#hyper parameters
	ep=.1
	lam=.9
	it=int(.5e5)
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
	    rew=int(np.sum([(nbot-st.index(i))*3 for i in ons])*pwr/fulp[dep]) 
	  else: rew=0
	  st1=''.join(st1)
	  if prin: print('st0 %s, st1 %s, ON %s, reward %d'%(st,st1,ons,rew))
	  st1=res.index(st1)
	  return rew,st1,ons

	#train
	st=0
	sts=[]; acts=[]; onrobstr=[]
	qtab=np.ones((spaln,actln),dtype='uint8')
	qplu=[]
	for i in range(actln):
	  bn=bin(i)[2:]
	  oncn=len([j for j in bn if j=='1'])
	  if oncn>nbot//2: qplu.append(i)
	#qtab[:,qplu]=2
	for i in range(it):
	  if not i%(it//20): print(int(i/it*100),end=' ')	
	  mx=np.argmax(qtab[st,:])
	  act=np.random.choice([mx,0],p=[ep,1-ep])
	  if act==0: act=np.random.choice(actln)
	  rew,st1,ons=reward(st,act,dep,prin=0)
	  qtab[st,act]=qtab[st,act]*(1-lam) + lam*rew
	  st=st1
	  sts.append(st)
	  acts.append(act)
	  onrobstr.append(len(ons))

	#play
	st=0
	sts=[];	acts=[]; onrobs=[]
	for i in range(1000):
	  act=np.argmax(qtab[st,:])
	  rew,st1,ons=reward(st,act,dep,prin=1)
	  st=st1
	  sts.append(st)
	  acts.append(act)
	  onrobs.append(ons)
	
	#plot
	win=1000
	ln=len(onrobstr)
	smo=np.zeros(ln-win)
	for i in range(ln-win):
	  smo[i]=np.mean(onrobstr[i:i+win])
	sim=[len(k) for k in onrobs]
	fig,ax=plt.subplots(1,2)
	ax[0].plot(smo)
	ax[1].plot(sim)  
	ax[0].set_title('learning curve')
	ax[1].set_title('no. of active bots')
	ax[0].set_xlabel('iteration')
	ax[1].set_xlabel('simulation time')
	ax[0].set_ylabel('active robots')
	ax[1].set_ylabel('active robots')	
	#plt.show()
	plt.savefig('output.jpg')
	#print(sim)
	return onrobs

window = Tk()
window.title("Life on Mars 2021")
window.geometry('400x400')
c=Canvas(window, width=400, height=400)
#c.pack_forget()
#onrobs=calc()	
app = Application(c)
window.mainloop()

