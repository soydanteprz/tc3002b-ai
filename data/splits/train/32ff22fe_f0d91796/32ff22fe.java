import java.io.*;
import java.util.*;
public class MyClass {
    public static void pA(int[]a){
        for(int i=0;i<a.length;i++){
            System.out.print(a[i]+" ");
        }
        System.out.println();
    }
    public static void main(String args[])throws IOException {
        BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
        int t=Integer.parseInt(br.readLine());
        while(t-->0){
            int n=Integer.parseInt(br.readLine());
            String s[]=br.readLine().split(" ");
            int arr[]=new int[n];
            int a[]=new int[n];
            for(int i=0;i<n;i++) arr[i]=Integer.parseInt(s[i]);
            if(n%2==0){
                for(int i=0;i<n-1;i+=2){
                    a[i]=arr[i+1];
                    a[i+1]=-1*arr[i];
                }
            }
            else{
                if(arr[1]+arr[2]!=0){
                    a[0]=arr[1]+arr[2];
                    a[1]=-1*arr[0];
                    a[2]=-1*arr[0];
                }
                else if(arr[0]+arr[1]!=0){
                    a[2]=arr[0]+arr[1];
                    a[0]=-1*arr[2];
                    a[1]=-1*arr[2];
                }
                else{
                    a[1]=arr[0]+arr[2];
                    a[0]=-1*arr[1];
                    a[2]=-1*arr[1];
                }
                for(int i=3;i<n-1;i+=2){
                    a[i]=arr[i+1];
                    a[i+1]=-1*arr[i];
                }
            }
            pA(a);
        }
    }
}
