import java.util.*;
import java.io.*;
public class Main  {
    public static void main(String args[]) {
      FastScanner fs=new FastScanner();    
      PrintWriter out=new PrintWriter(System.out);
      
      int t=fs.nextInt();
      while(t-->0)
      {
          int n=fs.nextInt();
          int c[]=fs.readArray(n);
          int i;
          
          
          long mn0=c[0];
          long mn1=c[1];
          long s0=c[0];
          long s1=c[1];
          
          
          long ans;
          ans=(mn0+mn1)*n*1L;
          
          for(i=2;i<n;i++)
          {
              if(i%2==0)
              {
                  s0+=c[i];
                  mn0=Math.min(mn0,c[i]);
              }
              else
              {
                  s1+=c[i];
                  mn1=Math.min(mn1,c[i]);
              }
              ans=Math.min(ans,s0+mn0*(n-i/2-1)+s1+mn1*(n-(i+1)/2));
          }
          
          out.println(ans);
          
          
          
          
      }
        
        out.flush();
        out.close();
    }
	
}
        
