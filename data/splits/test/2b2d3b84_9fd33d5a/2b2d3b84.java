import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;
import java.util.StringTokenizer;
public class Main
{



public static void main(String[] args)
{
   FastReader in = new FastReader();
   int t=in.nextInt();
   while(t-->0)
   {
       int n=in.nextInt();
       int k=in.nextInt();
       
       int a[]=new int[k];
       int ans[]=new int[n];
       int tem[]=new int[k];
       
       for(int i=0;i<k;i++)
           a[i]=in.nextInt();
       for(int  i=0;i<k;i++)
           tem[i]=in.nextInt();
       
      long c[]=new long[n];
      long l[]=new long[n];
      long r[]=new long[n];
      Arrays.fill(c,Integer.MAX_VALUE);
      Arrays.fill(l, Integer.MAX_VALUE);
      Arrays.fill(r,Integer.MAX_VALUE);
      
      long p=Integer.MAX_VALUE;
      for(int i=0;i<k;i++)
          c[a[i]-1]=tem[i];
      
      for(int i=0;i<n;i++)
      {
         p=Math.min(p+1,c[i]);
         l[i]=p;
      }
      p=Integer.MAX_VALUE;
      for(int i=n-1;i>=0;i--)
      {
         p=Math.min(p+1,c[i]);
         r[i]=p;
      }
    
       
      for(int i=0;i<n;i++)
          System.out.print(Math.min(l[i],r[i])+" ");
      System.out.println();
   }
}
}
