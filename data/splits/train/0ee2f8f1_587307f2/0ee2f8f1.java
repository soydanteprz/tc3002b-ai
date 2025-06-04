import java.util.*;
public class Main{
  public static void main(String args[]){
    Scanner sc=new Scanner(System.in);
    int T=sc.nextInt();
    while(T-->0){
     int n=sc.nextInt();
     int arr[]=new int[n];
     int min=Integer.MAX_VALUE;int max=Integer.MIN_VALUE;
     for(int i=0;i<n;i++){
      arr[i]=sc.nextInt();
      min=Math.min(arr[i],min);
      max=Math.max(arr[i],max);
     }
     while(min<=max){
      int mid=min+(max-min)/2;
      if(helper(arr,mid))
       min=mid+1;
       else max=mid-1;
     }
     System.out.println(min-1);
    }
  }
  public static boolean helper(int arr[],int min){
   int tmp[]=Arrays.copyOf(arr,arr.length);
   for(int i=arr.length-1;i>=2;i--){
    if(tmp[i]<min)
     return false;
    int d=(Math.min(arr[i],tmp[i]-min))/3;
    tmp[i-1]+=d;
    tmp[i-2]+=d*2;
   }
   return tmp[1]>=min && tmp[0]>=min;
  }
}
