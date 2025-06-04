import javax.swing.plaf.IconUIResource;
import java.lang.reflect.Array;
import java.text.CollationElementIterator;
import java.util.*;
import java.io.*;

//Timus judge id- 323935JJ


public class Main {




    //----------------------------------------------------------------------------------------------
    


    public static int mod = (int) (1e9 + 7);
    static int ans = Integer.MAX_VALUE;

    public static void main(String hi[]) throws Exception {
        FastReader sc = new FastReader();
        int t =sc.nextInt();
        while(t-->0)
        {
            int n =sc.nextInt();
            String a = sc.nextLine(),b=sc.nextLine();

            int count1=0,count2=0,count3=0,count4=0;

            for(int i=0;i<n;i++)
            {
                if(a.charAt(i)=='0'&&b.charAt(i)=='0')
                    count1++;
                else  if(a.charAt(i)=='1'&&b.charAt(i)=='1')
                    count2++;
                else  if(a.charAt(i)=='1'&&b.charAt(i)=='0')
                    count3++;
                else  if(a.charAt(i)=='0'&&b.charAt(i)=='1')
                    count4++;
            }
            int ans=Integer.MAX_VALUE;

            if(count3==count4)
                ans=Math.min(count3*2,ans);

            if(count2==count1+1)
                ans=Math.min(ans,2*count1+1);

            if(count2==1&&count1==0)
                ans=1;
            if(count3==0&&count4==0)
                ans=0;
            if(ans==Integer.MAX_VALUE)
                System.out.println(-1);
            else System.out.println(ans);

        }

    }

    





    

    // method to return LCM of two numbers
    


    

    // method to return LCM of two numbers
    


    
}
