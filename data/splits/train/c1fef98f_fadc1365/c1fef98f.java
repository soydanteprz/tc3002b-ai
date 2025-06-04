import java.util.*;
import java.io.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Scanner;
import java.util.StringTokenizer;

public class Test{

        static FastReader scan;

        static void solve(){
            scan.nextLine();
            int n=scan.nextInt();
            int k=scan.nextInt();
            int []arr=new int[n+1];

            int []block=new int[k];
            int []temp=new int[k];

            for(int i=0;i<k;i++){
                block[i]=scan.nextInt();
            }

            for(int i=0;i<k;i++){
                temp[i]=scan.nextInt();
            }

            for(int i=0;i<k;i++){
                arr[block[i]]=temp[i];
            }
            int []dp=new  int[n+1];
            int prev=Integer.MAX_VALUE-1000;
            for(int i=1;i<=n;i++){
                if(arr[i]==0){
                    if(prev==Integer.MAX_VALUE-1000){
                        dp[i]=prev;
                    }
                    else{
                        prev=prev+1;
                        dp[i]=prev;
                    }
                }
                else{
                    prev=Math.min(prev+1,arr[i]);
                    dp[i]=prev;
                }
            }

            prev=Integer.MAX_VALUE-1000;
            for(int i=n;i>=1;i--){
                if(arr[i]==0){
                    if(prev==Integer.MAX_VALUE-1000){
                        dp[i]=Math.min(prev,dp[i]);
                    }
                    else{
                        prev=prev+1;
                        dp[i]=Math.min(prev,dp[i]);
                    }
                }
                else{
                    prev=Math.min(prev+1,arr[i]);
                    dp[i]=Math.min(prev,dp[i]);
                }
            }

            for(int i=1;i<=n;i++){
                System.out.print(dp[i]+" ");
            }
            System.out.println();

        }

        public static void main (String[] args) throws java.lang.Exception{
            scan=new FastReader();
            int t=scan.nextInt();
            while(t-->0){
                solve();                  
            }
        }

        

                

        

        

        

        

           

        

        

        

        
        

        
}
