import java.util.*;
import java.lang.*;
import java.io.*;




public class Codechef
{
    
       
     
    public static void main (String[] args) throws java.lang.Exception
    {
 
       try{
        FastReader sc=new FastReader();
        BufferedWriter out = new BufferedWriter(
            new OutputStreamWriter(System.out));
        int t=sc.nextInt();
        while(t-->0)
        {
            
            int n = sc.nextInt();
            
            int k[] = new int[n];
            for(int i=0;i<n;i++)
            {
                k[i] = sc.nextInt();
            }
            int h[] = new int[n];
            for(int i=0;i<n;i++)
            {
                h[i] = sc.nextInt();
            }
            
            
        long tail = k[n-1];
        long span = h[n-1];
            
            long ans = 0;
            
            for(int i=n-2;i>=0;i--)
            {
                if((tail-span)+1>k[i])
                {
                    ans+=(span*(span+1))/2;
                    tail=k[i];
                    span=h[i];
                }
                else if((tail-span)+1<=((k[i]-h[i])+1))
                {
                    continue;
                }
                else
                {
                    span+=(((tail-span)+1) - ((k[i]-h[i])+1));
                }
            }
            
            ans+=(span*(span+1))/2;
            
            System.out.println(ans);
            
        }
    }catch(Exception e){
            return;
        }
    }
}
