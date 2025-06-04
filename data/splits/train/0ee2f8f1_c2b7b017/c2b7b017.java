import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

//
public class Example {
    

    public static void main(String[] args) {
        FastReader sc = new FastReader();

        int t= sc.nextInt();
        while(t>0){
            t--;
            int ans=Integer.MAX_VALUE;
            int n=sc.nextInt();
            int[] ar= new int[n];
            int l=Integer.MAX_VALUE;
            int h=Integer.MIN_VALUE;
            for(int i=0;i<n;i++){
                ar[i]=sc.nextInt();
                l=Math.min(l,ar[i]);
                h=Math.max(h,ar[i]);

            }
            int[] extra;
            while(l<=h){
                int mid=l+(h-l)/2;
                if(possibleans(ar,mid)){
                    ans=mid;
                    l=mid+1;
                }else{
                    h=mid-1;
                }
            }
            System.out.println(ans);


        }
    }

    private static boolean possibleans(int[] ar, int mid) {
int[] extra=new int[ar.length];
        for(int i=ar.length-1;i>=2;i--){
            if((ar[i]+extra[i]-mid)<0){
                return false;
            }
            int d=Math.min(ar[i],extra[i]+ar[i]-mid);
            extra[i-1]=extra[i-1]+d/3;
            extra[i-2]+=2*(d/3);

        }
        int a=ar[0]+extra[0];
        int b=ar[1]+extra[1];
        return (a>=mid && b>=mid);

    }

    
}
