import java.io.*;
import java.util.*;



import java.math.*;
import java.math.BigInteger;


public final class A
{    
	static PrintWriter out = new PrintWriter(System.out); 
	static StringBuilder ans=new StringBuilder();
	static FastReader in=new FastReader();
	//	static node1 seg[];
	//static pair moves[]= {new pair(-1,0),new pair(1,0), new pair(0,-1), new pair(0,1)};
	public static void main(String args[])throws IOException
	{				


	
		int T=i();
		outer:while(T-->0)
		{
			int N=i();
			int size[]=new int[N];
			PriorityQueue<node1> q[]=new PriorityQueue[6];
			for(int i=0; i<6; i++)q[i]=new PriorityQueue<node1>();
			for(int i=0; i<N; i++)
			{
				char X[]=in.next().toCharArray();
				int s=X.length;
				size[i]=s;
				int f[]=new int[6];
				for(char x:X)f[x-'a']++;
				for(int j=0; j<6; j++)q[j].add(new node1(f[j],i,s));
			}
			
			int max=0;
			for(int i=0; i<6; i++)
			{
				PriorityQueue<node1> q_new=new PriorityQueue<>();
				q_new=q[i];
				int c=0;
				long f=0;
				while(q_new.size()>0)
				{
					node1 x=q_new.remove();
//					System.out.println(x.f+" "+x.size+" "+x.a);
					f+=x.a;
					if(f>0)
					{
						c++;
						max=Math.max(max, c);
					}
					else break;
				}
			}
			out.println(max);
		}
		out.close();

	}
	
	
	

	
	
	
	


	
	
	//	static void build(int v,int tl,int tr,long A[])
	//	{
	//		if(tl==tr)
	//		{
	//			seg[v]=new node1(A[tl],A[tr],1,true);
	//			return ;
	//		}
	//		int tm=(tl+tr)/2;
	//		build(2*v,tl,tm,A);
	//		build(2*v+1,tm+1,tr,A);
	//		seg[v]=merge(seg[2*v],seg[2*v+1]);
	//	}
	//	static node1 ask(int v,int tl,int tr,int l,int r)
	//	{
	//		if(l>r)return new node1(0,0,0,false);//verify true or false
	//		if(tl==l && tr==r)return seg[v];
	//		int tm=(tl+tr)/2;
	//		node1 a=ask(v*2,tl,tm,l,Math.min(tm, r));
	//		node1 b=ask(v*2+1,tm+1,tr,Math.max(tm+1, l),r);
	//		return merge(a,b);
	//	}
	//	static node1 merge(node1 a,node1 b)
	//	{
	//		long s=0;
	//		long l1=a.L,r1=a.R,c1=a.cnt;
	//		long l2=b.L,r2=b.R,c2=b.cnt;
	//		long g=GCD(l2,r1); s=c1+c2;
	//		if(g==1)
	//		{
	//			s--;
	//			g=(l2*r1)/g;
	//			if(c1==1)
	//			{
	//				l1=g;
	//			}
	//			if(c2==1)r2=g;
	//			return new node1(l1,r2,s,true);
	//		}
	//		return new node1(l1,r2,s,a.leaf^b.leaf);
	//	}
	
	
	
	
	
	



	
	
	



	





	
	
	


	
	
	

	    



	 

	

	

	

	  

	



	

	

	

	

	

	

	
	
	
	
	
	

	static int i()
	{
		return in.nextInt();
	}

	

	

	

	

}
class node1 implements Comparable<node1>
{
	int index,f,size;
	long a;
	node1(int f,int i,int size)
	{
		this.f=f;
		this.index=i;
		this.size=size;
		a=2*f-size;
	}
	
}


//Code For FastReader
//Code For FastReader
//Code For FastReader
//Code For FastReader

