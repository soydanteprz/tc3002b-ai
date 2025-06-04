import java.util.Scanner;
import java.util.Arrays;
public class B {
    public static int[] sort(int[] arr,int low,int high){
        int[] sortedArray;
        if (low==high) {
            sortedArray=new int[]{arr[low]};
            return sortedArray;
        }
        int mid=(low+high)/2;
        int[] arr1=sort(arr,low,mid);
        int[] arr2=sort(arr,mid+1,high);

        sortedArray=merge(arr1,low,mid,arr2,mid+1,high);
        return sortedArray;

    }

    public static int[] merge(int[] arr1,int low1,int high1,int[] arr2,int low2,int high2){

        int size1=high1-low1+1;
        int size2=high2-low2+1;
        
        int[] arr=new int[size1+size2];
        int pointer=0;
        low1=0;low2=0;
        while (low1<size1 && low2<size2){
            if (arr1[low1] <= arr2[low2]) {
                arr[pointer++]=arr1[low1++];
            }
            else{
                arr[pointer++]=arr2[low2++];
            }

        }

        while(low1<size1) arr[pointer++]=arr1[low1++];
        while(low2<size2) arr[pointer++]=arr2[low2++];
        return arr;
    }
    
    public static void main(String[] args){
        Scanner sc=new Scanner(System.in);
        int t=sc.nextInt();
        while(t-- > 0){
            int n=sc.nextInt();
            long k=sc.nextLong();
            int[] arr=new int[n];
            long sum=0;
            for(int i=0;i<n;i++) {
                arr[i]=sc.nextInt();
                sum+=arr[i];
            }
            if (sum<=k){
                System.out.println(0);
                continue;
            }
            long s[]=new long[n];
            arr=sort(arr,0,n-1);
            s[n-1]=arr[n-1]-arr[0];
            long ans=sum-k;
        
            for(int i=n-2;i>=0;i--) s[i]=s[i+1]+arr[i]-arr[0];
           
            for(int i=n-1;i>=1;i--) {
                
                long caseAns=(sum-k-s[i]+n-i)/(n-i+1);
                
                caseAns=Math.max(caseAns,0);
                caseAns+=n-i;
                ans=Math.min(caseAns,ans);
            }
            System.out.println(ans);
        }
        sc.close();
    }
}
