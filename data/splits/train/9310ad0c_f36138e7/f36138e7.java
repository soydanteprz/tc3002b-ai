import java.io.*;
import java.util.*;

public class a {

    public static void main(String[] args){

        FastScanner sc = new FastScanner();
        int t = sc.nextInt();
        while(t-- > 0){

            int n = sc.nextInt();

            long time[] = new long[n];
            long health[] = new long[n];

            for(int i=0; i<n; i++){
                long temp = sc.nextLong();
                time[i] = temp;
            }
            for(int i=0; i<n; i++){
                long temp = sc.nextLong();
                health[i] = temp;
            }

            long ans = 0L;
            int peak = n-1;
            int end = n-1;
            for(int i=n-1; i>=0; i--){

                if(i == n-1){
                    ans += (health[i]*(health[i]+1))/2;
                    continue;
                }

                if(health[peak] - (time[peak]-time[i]) <= 0){
                    peak = i;
                    end = i;
                    ans += (health[i]*(health[i]+1))/2;
                }

                else if(health[peak] - (time[peak]-time[i]) < health[i]){

                    long val = health[peak] + (time[end]-time[peak]); 
                    ans -= (val*(val+1))/2;

                    //System.out.println("Val1: " + val);

                    val = health[i] + (time[end]-time[i]);
                    ans += (val*(val+1))/2;

                    //System.out.println("Val2: " + val);

                    peak = i;
                }

            }

            System.out.println(ans);

        }

    }

}


