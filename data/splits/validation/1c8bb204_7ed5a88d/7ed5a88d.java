import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.io.PrintStream;
            //import java.util.*;

        public class Solution {

            public static final boolean LOCAL = System.getProperty("ONLINE_JUDGE")==null;

            
                                    
            

        
        public static void main(String[] args) throws Exception {
            
            FastScanner s = new FastScanner();

            PrintStream debug = null;
            
            if(LOCAL){
                s = new FastScanner("src/input.txt");
                PrintStream o = new PrintStream("src/sampleout.txt");
                debug = new PrintStream("src/debug.txt");
                System.setOut(o);
            }

            


            int tcr = s.nextInt();

            
            for(int tc=0;tc<tcr;tc++){
                int n = s.nextInt();

                long arr[] = new long[n];

                for(int i=0;i<n;i++){
                    arr[i] = s.nextLong();
                }

                long min1 = arr[0];
                long min2 = arr[1];
                long sum1 = arr[0];
                long sum2 = arr[1];

                int cnt1 = 1;
                int cnt2 = 1;

                long ans = arr[0]*(n) + arr[1]*(n);

                for(int i=2;i<n;i++){
                    if((i % 2) == 0){
                        min1 = Math.min(min1,arr[i]);
                        sum1 += arr[i];
                        cnt1++;
                    }else{
                        min2 = Math.min(min2,arr[i]);
                        sum2 += arr[i];
                        cnt2++;
                    }
                    long temp = min1*(n - cnt1 + 1) + (sum1 - min1) + (min2*(n- cnt2 + 1) + (sum2 - min2));
                    ans = Math.min(ans,temp);
                }

                println(ans);
            }                
        }

        

        


        

       
  

        

        

        

        

        // return the (index + 1)
        // where index is the pos of just smaller element
        // i.e count of elemets strictly less than num
        
        
        //return (index of just greater element)
        //count of elements smaller than or equal to num

        
    
        

        public static void println(Object obj){
            System.out.println(obj.toString());
        }

        

        

        

        

    

        

        


        

        // Merge sort function
        

        

}
