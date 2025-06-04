import java.util.*;
import java.lang.*;
import java.io.*;
public class MyAnswer {
     public static void main(String[] args)throws IOException{
          FastScanner scan = new FastScanner();
          //SuperFastScanner scan = new SuperFastScanner();
          PrintWriter out = new PrintWriter(System.out);
          StringBuilder result = new StringBuilder();
     
          int t = scan.nextInt();
          while (t-- > 0){
               int n = scan.nextInt();
               int arr[] = scan.nextIntArray(n);
               if(n%2==0){
                    int j = n/2;
                    for(int i = j;i<n;i++){
                         int val = arr[i] * -1;
                         result.append(val + " ");
                    }
                    for(int i = 0;i<j;i++){
                         int val = arr[i] ;
                         result.append(val + " ");
                    }
               }else{
                    int j = (n-3)/2;
                    for(int i = j;i<n-3;i++){
                         int val = arr[i] * -1;
                         result.append(val + " ");
                    }
                    for(int i = 0;i<j;i++){
                         int val = arr[i] ;
                         result.append(val + " ");
                    }
                    
                    if(arr[n-3]+arr[n-2] !=0 ){
                         int sum = arr[n-3] + arr[n-2];
                         sum *=-1;
                         result.append(arr[n-1] + " ");
                         result.append(arr[n-1] + " ");
                         result.append(sum+ " ");
                         
                    }else if(arr[n-1]+arr[n-2] !=0 ){
                         int sum = arr[n-1] + arr[n-2];
                         sum *=-1;
                         result.append(sum+ " ");
                         result.append(arr[n-3] + " ");
                         result.append(arr[n-3]+" ");
                         
                    }else{
                         int sum = arr[n-1] + arr[n-3];
                         sum *=-1;
                         result.append(arr[n-2] + " ");
                         result.append(sum+ " ");
                         result.append(arr[n-2]+ " ");
                    }
                         
               }
               result.append("\n");
               
          }
          out.println(result);
          out.flush();
     }
     

     
     

}
