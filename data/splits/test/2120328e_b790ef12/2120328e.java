import java.util.*;
import java.lang.*;
import java.io.*;
public class Main  
{
    static FastReader sc =new FastReader();
    static PrintWriter out=new PrintWriter(System.out);
    static int mod=10000007;
   static StringBuilder sb=new StringBuilder();
    /* start */
       public static void main(String [] args)
    {
            int t = i();
            while(t-->0)
            {
                int n = i();
                int a[] = input(n);
                char c[] = inputC();
                ArrayList<Integer> b = new ArrayList<>();
                ArrayList<Integer> r = new ArrayList<>();
                for(int i=0;i<n;i++)
                {
                    if(c[i]=='R')
                    r.add(a[i]);
                    else 
                    b.add(a[i]);
                }
                Collections.sort(b);
                Collections.sort(r,Collections.reverseOrder());
                boolean is = true;
                int cnt = 1;
                for(int i=0;i<b.size();i++)
                {
                    if(b.get(i)<cnt)
                    {
                        is = false;
                        break;
                    }
                    cnt++;
                }
                for(int i=0;i<r.size();i++)
                {
                    if(r.get(i)>n-i)
                    {
                        is = false;
                        break;
                    }
                }

                out.println(is==true?"YES":"NO");
            }
            out.close();
    }

    
    /* end */
     

    
    static int i() {
        return sc.nextInt();
    }

    static String s() {
        return sc.next();
    }

    

    static char[] inputC()
    {
        String s = sc.nextLine();
        return s.toCharArray();
    }

    static int[] input(int n) {
        int A[]=new int[n];
           for(int i=0;i<n;i++) {
               A[i]=sc.nextInt();
           }
        return A;
    }

    

    

    
      
     

      

     
    
     

     
    
    

    
    
    
    
    
    
    
    
    

    

    

    

    

    

    

     

      
        //pair class
        
}
