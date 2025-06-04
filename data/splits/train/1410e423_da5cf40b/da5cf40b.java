import java.util.*;
import java.lang.*;
import java.io.*;
public class Main  
{
    static FastReader sc =new FastReader();
    static PrintWriter out=new PrintWriter(System.out);
    /* start */
   
       public static void main(String [] args)
    {
        // int testcases = 1;
        int testcases = i();
        while(testcases-->0)
        {
            
            solve();
        }
        out.flush();
        out.close();
    }
    static void solve()
    { 
        int n = i();
        char c[] = inputC();
        char d[] = inputC();
        int x01=0,x10=0,x00=0,x11=0;
        for(int i=0;i<n;i++)
        {
            if(c[i]=='0'&&d[i]=='0')x00++;
            if(c[i]=='0'&&d[i]=='1')x01++;
            if(c[i]=='1'&&d[i]=='0')x10++;
            if(c[i]=='1'&&d[i]=='1')x11++;
        }

        int ans = Integer.MAX_VALUE;
        if(x01==0 && x10==0)
        {
            System.out.println(0);
            return ;
        }
        if(x11==x00+1)
        {
            ans = Math.min(x11+x00,ans);
        }
      if(x01==x10)
        {
            ans = Math.min(x01+x10,ans);
        }
        if(ans == Integer.MAX_VALUE){
            ans = -1;
        }
        System.out.println(ans);
    }
    
    /* end */
     

    static int i() {
        return sc.nextInt();
    }


    

    static char[] inputC()
    {
        String s = sc.nextLine();
        return s.toCharArray();
    }

    

    

    

    
      
     

      

     
    
     

    
    

    

    

    

    

    

    

     

      

        
    
        
        //pair class
        
}
