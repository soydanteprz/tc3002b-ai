import java.io.BufferedReader; 
import java.io.IOException; 
import java.io.InputStreamReader; 
import java.util.*;
  
public class aa
{ 
     
   
    public static void main(String[] args) 
    {
            FastReader d=new FastReader();
            int t,i,j,c,z,k,l,n;
            int mod = (int) 1e9 + 7;
            int Inf=Integer.MAX_VALUE;
            int negInf=Integer.MIN_VALUE;
        t=d.nextInt();
        //t=1;
        //String s;
        //char ch,ch1,ch2,ch3;
        while(t-->0)
        {
            z=c=0;
            n=d.nextInt();
            int a[]=new int[n];
            for(i=0;i<n;i++)
            a[i]=d.nextInt();
            //s=d.nextLine();//dont need extra d.nextLine()
            long p=0;
            long ans;
            long x,y;
            long e,o;
            ans=Long.MAX_VALUE;
            x=y=Integer.MAX_VALUE;
            e=o=n;
            for(i=0;i<n;i++) {
                if(i%2==1) {
               	p+=a[i];
               	e--;
               	x=Long.min(x, a[i]);
               	ans=Long.min(ans,p+(o*y)+(e*x));
               	}
               	else {
               	p+=a[i];
               	o--;
               	y=Long.min(y, a[i]);
               	ans=Long.min(ans,p+(o*y)+(e*x));
                	}
                }
                System.out.println(ans);
        }
    } 
}
