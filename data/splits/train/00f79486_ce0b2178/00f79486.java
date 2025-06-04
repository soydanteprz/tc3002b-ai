import java.util.*;

import java.io.*;  
public class Main
{
	
    PrintWriter out;
    FastReader sc;
    StringBuilder sb;
   
    /******************************************************************************************
     *****************************************************************************************/
    
	public void sol() {
		int n=ni();
		int[] ar=new int[n];
		for(int i=0;i<n;i++)ar[i]=ni();
		char[] s=rl();
		PriorityQueue<Integer> red=new PriorityQueue<>(Collections.reverseOrder()),blue=new PriorityQueue<>();
		for(int i=0;i<n;i++) {
			if(s[i]=='R')red.add(ar[i]);
			else blue.add(ar[i]);
		}
		int a=n;
		while(red.size()>0) {
			if(red.poll()>a) {
				no();return;
			}a--;
		}a=1;
		while(blue.size()>0) {
			if(blue.poll()<a) {
				no();return;
			}a++;
		}yes();
	}
    public static void main(String[] args) 
	{ 
	    Main g=new Main();
		g.out=new PrintWriter(System.out);
		g.sc=new FastReader();
		int t=1;
 		t=g.ni();
		while(t-->0)
		g.sol();
		g.out.flush();
	}
	
	/****************************************************************************************
     *****************************************************************************************/
	 public int ni(){
	      return sc.nextInt();
	}public char[] rl(){
	      return sc.nextLine().toCharArray();
	}
	public void pl(Object s){
	      out.println(s);
	}
	public String next(){
	      return sc.next();
	}
	
	
      
      
       
	 
	 
	
	
	
	
      
      void yes(){
            pl("YES");
      }void no(){
            pl("NO");
      }
      
    
     
    
    
    
      
	 
}
