import java.util.*;
import java.io.*;

public class hmm {

	static Scanner sc = new Scanner(System.in);
	static PrintWriter pw = new PrintWriter(System.out);


	public static void main(String[] args) throws Exception {
		int t =sc.nextInt();
		while (t-- > 0) {
             int n = sc.nextInt();
             int []k = sc.nextIntArray(n);
             int h[]=sc.nextIntArray(n);
     
             long mana = 0;
             pair cur = new pair(k[n-1]-h[n-1]+1,k[n-1]);
             for(int i=n-1;i>=0;i--) {
            	 int s = k[i];
            	 int start = s-h[i]+1;
            	 if(s>=cur.x) {
            		 cur.x = Math.min(start, cur.x);
            	 }
            	 else {
            		 long x = cur.y - cur.x +1;
            		 mana += x*(x+1)/2;
            		 cur.x = start;
            		 cur.y = s;
            	 }
             }
       
             long x = cur.y - cur.x +1;
       
    		 mana += x*(x+1)/2;
    		 pw.println(mana);
             
           
		}

		pw.close();

	}

	// -------------- stuff ------------------------------
	static class pair {
		int x ;
		int y;
	
		
		public pair(int n,int c) {
			x= n;
			y = c;
		
			
		}
	
	
		
	}

	

}
