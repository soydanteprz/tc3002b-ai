import java.util.*;
import java.io.*;
import java.math.BigInteger;
 
public class code{
 
    
    
    
    //@SuppressWarnings("unchecked")
    public static void main(String[] arg) throws IOException{
        //Reader in=new Reader();
        
        PrintWriter out = new PrintWriter(System.out);
        Scanner in = new Scanner(System.in);
        
        int t=in.nextInt();
        
        while(t-- > 0){
          int n=in.nextInt();
          long k=in.nextLong();

          PriorityQueue<Long> pq=new PriorityQueue<Long>();
          long[] sum=new long[n+1];

          for(int i=0;i<n;i++){
            long a=in.nextLong();
            pq.offer(a);
            //sum[i+1]=arr[i]+sum[i];
          }

          
          for(int i=1;i<=n;i++){
            sum[i]=pq.poll()+sum[i-1];
          }

          long min=(long)1e18;
          for(int i=0;i<n;i++){
            long x=sum[1]-floor(k-sum[n-i]+sum[1],(long)i+1);
            min=Math.min(min,(long)i+Math.max(0L,x));
          }

          out.println(min);
        }
 
        out.flush();
    }

    public static long floor(long a,long b){
      long z=a/b;
      if(b*z>a) z--;
      return z;
    }
 
}
